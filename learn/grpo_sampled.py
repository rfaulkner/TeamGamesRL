# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sampled (TRL-based) GRPO runner.

This module implements the *sampled* GRPO variant that collects prompts
by playing episodes, then trains via TRL's ``GRPOTrainer``.  Suitable
for larger games where exhaustive game-tree enumeration is infeasible.

All public functions accept a ``runner`` parameter — the ``GRPORunner``
instance that holds shared state (env, backend, config, callbacks).
"""

import contextlib
import copy
import dataclasses
import os
import time
import zlib

from absl import logging
from learn.trajectory import PlayerTrajectory
from learn.trajectory import RLTrajectoryStep
import numpy as np
import torch

try:
  import pyspiel
except ImportError:
  pyspiel = None  # HLE adapter used for Hanabi instead


def _serialize_game_and_state(game, state):
  """Serialize game+state, dispatching to adapter or pyspiel."""
  from env.hanabi.hanabi_env import HanabiGame  # pylint: disable=g-import-not-at-top
  from env.hanabi import hanabi_env  # pylint: disable=g-import-not-at-top

  if isinstance(game, HanabiGame):
    return hanabi_env.serialize_game_and_state(game, state)
  return pyspiel.serialize_game_and_state(game, state)


def _deserialize_game_and_state(data_str):
  """Deserialize game+state, dispatching to adapter or pyspiel."""
  import json as _json  # pylint: disable=g-import-not-at-top

  try:
    data = _json.loads(data_str)
  except (ValueError, TypeError):
    # Not JSON — fall through to pyspiel.
    return pyspiel.deserialize_game_and_state(data_str)
  if isinstance(data, dict) and data.get('adapter') == 'hanabi_env':
    from env.hanabi import hanabi_env  # pylint: disable=g-import-not-at-top

    return hanabi_env.deserialize_game_and_state(data_str)
  return pyspiel.deserialize_game_and_state(data_str)


# ═══════════════════════════════════════════════════════════════════════
# Prompt collection
# ═══════════════════════════════════════════════════════════════════════


def collect_game_prompts(
    runner,
    num_episodes: int,
    pass_idx: int = 1,
    start_time: float = 0.0,
) -> list[dict]:
  """Collect game-state prompts by playing episodes with LLM agents.

  Plays ``num_episodes`` games and records the prompts shown to each
  player at each decision point, along with the action history and
  player index needed to simulate game completion for reward
  computation.

  Args:
    runner: The ``GRPORunner`` instance.
    num_episodes: Number of episodes to play.
    pass_idx: The current GRPO pass index.
    start_time: Training start timestamp for elapsed time calculation.

  Returns:
    A list of prompt-entry dicts with keys: ``prompt``, ``player_id``,
    ``action_history``, ``legal_actions``, ``legal_actions_desc``,
    ``state_text``, ``serialized_state``.
  """
  all_prompts = []
  num_players = runner._game_config.num_players
  ep_rewards = []

  for ep in range(1, num_episodes + 1):
    time_step = runner._env.reset()
    action_history = []
    trajectories = [PlayerTrajectory(player_id=p) for p in range(num_players)]

    bot_player = None
    if getattr(runner._config, 'bot_partner', False) and num_players == 2:
      # Alternating roles: odd episodes -> P1 is bot; even episodes -> P0 is bot.
      bot_player = 1 if ep % 2 == 1 else 0

    while not time_step.last():
      current_player = time_step.current_player()
      state = runner._env._state  # pylint: disable=protected-access

      state_text = runner._renderers[current_player].render_state(
          state, current_player, runner._env.game
      )
      legal_actions_with_desc = runner._renderers[
          current_player
      ].render_legal_actions(state, current_player, runner._env.game)
      legal_actions = [a for a, _ in legal_actions_with_desc]
      action_descriptions = [d for _, d in legal_actions_with_desc]

      is_bot_turn = (bot_player is not None and current_player == bot_player)
      prompt = None
      response = None
      log_prob = 0.0

      if is_bot_turn:
        if not hasattr(runner, '_heuristic_bot') or runner._heuristic_bot is None:
          try:
            from env.hanabi.heuristic_player import SafePlayPlayer  # pylint: disable=g-import-not-at-top
            runner._heuristic_bot = SafePlayPlayer(seed=42)
          except ImportError:
            runner._heuristic_bot = None

        if runner._heuristic_bot is not None:
          action_id = runner._heuristic_bot.select_action(
              state, current_player, runner._env.game
          )
        else:
          action_id = int(np.random.choice(legal_actions))
        action_text = state.action_to_string(current_player, action_id)
        response = action_text
      else:
        prompt = runner._agents[current_player]._build_prompt(  # pylint: disable=protected-access
            state_text, legal_actions, action_descriptions
        )

        response, log_prob = runner._backend.generate_with_logprobs(
            prompt,
            temperature=runner._current_temperature,
            max_tokens=runner._config.max_completion_length,
        )
        action_id = runner._renderers[current_player].parse_action(
            response, legal_actions_with_desc
        )
        if action_id is None:
          action_id = int(np.random.choice(legal_actions))

        # Epsilon-greedy exploration: with probability epsilon, override the
        # model's action with a uniformly random legal action.  This prevents
        # game-ending action collapse (e.g. always playing cards in Hanabi)
        # and ensures longer, more diverse collection episodes.
        epsilon_explored = False
        if (
            runner._current_epsilon > 0
            and np.random.random() < runner._current_epsilon
        ):
          action_id = int(np.random.choice(legal_actions))
          epsilon_explored = True

        action_text = state.action_to_string(current_player, action_id)

        prompt_entry = {
            'prompt': prompt,
            'player_id': current_player,
            'action_history': list(action_history),
            'legal_actions': legal_actions,
            'legal_actions_desc': legal_actions_with_desc,
            'state_text': state_text,
            'serialized_state': _serialize_game_and_state(
                runner._env.game, state
            ),
        }
        all_prompts.append(prompt_entry)
        runner._prompt_metadata[prompt] = prompt_entry

      trajectories[current_player].steps.append(
          RLTrajectoryStep(
              prompt=prompt or '',
              action_text=response.strip() if response else '',
              action_id=action_id,
              log_prob=log_prob,
              state_text=state_text,
              llm_response=response or '',
              game_action_text=action_text,
          )
      )

      action_history.append(action_id)
      time_step = runner._env.step([action_id])

    if time_step.rewards is not None:
      for p in range(num_players):
        trajectories[p].reward = time_step.rewards[p]

    mean_r = float(np.mean([t.reward for t in trajectories]))
    ep_rewards.append(mean_r)

    global_ep = (pass_idx - 1) * num_episodes + ep
    if runner._log_episode_fn is not None:
      runner._log_episode_fn(global_ep, trajectories, 0.0, False)
    if runner._update_metrics_fn is not None:
      runner._update_metrics_fn(trajectories, 0.0)

    ep_elapsed = time.time() - start_time if start_time > 0 else 0.0
    actions_summary = ' | '.join(
        f'P{t.player_id}:[{",".join(s.game_action_text for s in t.steps)}]'
        for t in trajectories
    )
    role_info = ''
    if bot_player is not None:
      role_info = f' (P{1-bot_player}:LLM vs P{bot_player}:Bot)'
    print(
        f'[pass {pass_idx} collect {ep}/{num_episodes}]{role_info} reward={mean_r:.2f} '
        f'({ep_elapsed:.1f}s) {actions_summary}',
        flush=True,
    )

  mean_collected = float(np.mean(ep_rewards)) if ep_rewards else 0.0
  logging.info(
      'Pass %d collection complete: %d prompts from %d episodes '
      '(mean reward: %.3f)',
      pass_idx,
      len(all_prompts),
      num_episodes,
      mean_collected,
  )
  collect_stats = {
      'mean_reward': mean_collected,
      'min_reward': float(np.min(ep_rewards)) if ep_rewards else 0.0,
      'max_reward': float(np.max(ep_rewards)) if ep_rewards else 0.0,
      'std_reward': float(np.std(ep_rewards)) if ep_rewards else 0.0,
      'num_episodes': num_episodes,
      'num_prompts': len(all_prompts),
  }
  return all_prompts, collect_stats


# ═══════════════════════════════════════════════════════════════════════
# Reward simulation from serialized state
# ═══════════════════════════════════════════════════════════════════════


def simulate_from_state(
    runner,
    action_history: list[int],
    chosen_action: int,
    target_player: int,
    serialized_state: str | None = None,
) -> float:
  """Simulate a game to completion from a given state to get the reward.

  Restores the exact game state (preserving the original card deal) via
  ``serialized_state``, applies ``chosen_action``, then plays out the
  partner's remaining turns using frozen LoRA weights from the start of
  the current pass.

  Args:
    runner: The ``GRPORunner`` instance.
    action_history: List of action IDs taken before the current decision.
    chosen_action: The action to apply at the current decision point.
    target_player: The player whose reward we want.
    serialized_state: Serialized game-and-state string.

  Returns:
    The reward for ``target_player`` at the end of the simulated game.
  """
  # ── Restore the exact game state ──
  if serialized_state is not None:
    _, state = _deserialize_game_and_state(serialized_state)
    # _deserialize_game_and_state already returns a fresh clone for
    # Hanabi (in-memory cache) and a restored state for OpenSpiel.
    # No need for a second serialize/deserialize round-trip.
    runner._env.set_state(state)
  else:
    runner._env.reset()
    state = runner._env._state  # pylint: disable=protected-access
    for action_id in action_history:
      if state.is_terminal():
        break
      state.apply_action(action_id)

  # Apply the chosen action.
  state = runner._env._state  # pylint: disable=protected-access
  if not state.is_terminal():
    state.apply_action(chosen_action)

  sim_mode = runner._config.reward_simulation_mode
  horizon = runner._config.truncated_rollout_horizon

  if sim_mode == 'llm':
    # ── LLM-based playout (accurate but slow) ──
    _simulate_with_llm(runner, state, horizon)
  elif sim_mode == 'heuristic':
    # ── Heuristic playout (Hanabi-only, moderate speed) ──
    _simulate_with_heuristic(runner, state, horizon)
  elif sim_mode == 'rollout':
    # ── Random rollout + heuristic value (fast + decent signal) ──
    # Roll out randomly for `horizon` turns (default 6), then use a
    # game-specific heuristic to estimate the value of the resulting
    # state rather than playing all the way to terminal.
    rollout_depth = horizon if horizon is not None else 6
    _simulate_with_random(state, rollout_depth)
    if not state.is_terminal() and hasattr(state, 'state_value'):
      val = state.state_value()
      return val
  else:
    # ── Random playout (fast, ~1 ms per eval) ──
    _simulate_with_random(state, horizon)

  if state.is_terminal() and state.rewards() is not None:
    return float(state.rewards()[target_player])
  # For truncated rollouts, try to read intermediate score.
  if hasattr(state, 'state_value'):
    return state.state_value()
  if hasattr(state, 'returns'):
    try:
      return float(state.returns()[target_player])
    except (IndexError, TypeError):
      pass
  return 0.0


def _simulate_with_random(state, horizon: int | None = None) -> None:
  """Play out remaining turns with random legal actions.

  ~1 ms per game — no model inference needed.
  """
  turns_played = 0
  while not state.is_terminal():
    if horizon is not None and turns_played >= horizon:
      break
    player = state.current_player()
    legal = state.legal_actions(player)
    if not legal:
      break
    state.apply_action(int(np.random.choice(legal)))
    turns_played += 1


def _simulate_with_heuristic(runner, state, horizon: int | None = None) -> None:
  """Play out remaining turns with a rule-based heuristic player."""
  try:
    from env.hanabi.heuristic_player import SafePlayPlayer  # pylint: disable=g-import-not-at-top

    heuristic = SafePlayPlayer()
  except ImportError:
    logging.warning('Heuristic player unavailable — falling back to random.')
    _simulate_with_random(state, horizon)
    return

  game = getattr(runner._env, 'game', None)
  turns_played = 0
  while not state.is_terminal():
    if horizon is not None and turns_played >= horizon:
      break
    player = state.current_player()
    legal = state.legal_actions(player)
    if not legal:
      break
    action = heuristic.select_action(state, player, game)
    if action is None:
      action = int(np.random.choice(legal))
    state.apply_action(action)
    turns_played += 1


def _simulate_with_llm(runner, state, horizon: int | None = None) -> None:
  """Play out remaining turns using the LLM with frozen LoRA weights.

  Most accurate reward estimation but very slow (~18 sec per eval
  for a 12B model, since each remaining turn requires a full forward
  pass).
  """
  # ── Swap in frozen LoRA weights for partner simulation ──
  live_lora_state = None
  if runner._frozen_lora_state is not None:
    live_lora_state = copy.deepcopy({
        k: v
        for k, v in runner._backend.model.named_parameters()
        if v.requires_grad
    })
    for name, param in runner._backend.model.named_parameters():
      if name in runner._frozen_lora_state:
        param.data.copy_(runner._frozen_lora_state[name])

  try:
    turns_played = 0
    while not state.is_terminal():
      if horizon is not None and turns_played >= horizon:
        break
      current_player = state.current_player()

      state_text = runner._renderers[current_player].render_state(
          state, current_player, runner._env.game
      )
      legal_actions_with_desc = runner._renderers[
          current_player
      ].render_legal_actions(state, current_player, runner._env.game)
      legal_actions = [a for a, _ in legal_actions_with_desc]
      action_descriptions = [d for _, d in legal_actions_with_desc]

      prompt = runner._agents[current_player]._build_prompt(  # pylint: disable=protected-access
          state_text, legal_actions, action_descriptions
      )

      with torch.no_grad():
        response, _ = runner._backend.generate_with_logprobs(
            prompt,
            temperature=runner._current_temperature,
            max_tokens=runner._config.max_completion_length,
        )

      partner_action = runner._renderers[current_player].parse_action(
          response, legal_actions_with_desc
      )
      if partner_action is None:
        partner_action = (
            int(np.random.choice(legal_actions)) if legal_actions else 0
        )
      state.apply_action(partner_action)
      turns_played += 1
  finally:
    if live_lora_state is not None:
      for name, param in runner._backend.model.named_parameters():
        if name in live_lora_state:
          param.data.copy_(live_lora_state[name].data)


# ═══════════════════════════════════════════════════════════════════════
# Blended reward helpers
# ═══════════════════════════════════════════════════════════════════════


def _classify_hanabi_action_type(state, action_id: int, player_id: int) -> str:
  """Classify a Hanabi action as 'play', 'discard', or 'hint'.

  Args:
    state: The current game state.
    action_id: The action to classify.
    player_id: The acting player.

  Returns:
    One of 'play', 'discard', or 'hint'.
  """
  import re  # pylint: disable=g-import-not-at-top

  action_str = state.action_to_string(player_id, action_id)
  if re.match(r'\(Play \d+\)', action_str):
    return 'play'
  elif re.match(r'\(Discard \d+\)', action_str):
    return 'discard'
  else:
    return 'hint'


@contextlib.contextmanager
def _frozen_lora_active(runner):
  """Temporarily activate the frozen LoRA snapshot for policy inference.

  Partner / continuation moves must be sampled from a *stable* policy, not
  the one being updated mid-pass, so ``run_sampled`` snapshots the LoRA
  weights at the top of each pass into ``runner._frozen_lora_state``.  This
  context manager swaps that snapshot in and restores the live weights on
  exit.

  Reentrant: nested ``with`` blocks are no-ops, so a caller can wrap a whole
  multi-turn rollout without every inner sampling call re-cloning the
  adapter.  Swapping clones every trainable parameter, so an ``m``-turn
  continuation done naively would pay that cost ``m`` times.

  Yields:
    None.  The frozen weights are active for the duration of the block.
  """
  depth = getattr(runner, '_frozen_lora_depth', 0)
  if depth > 0 or runner._frozen_lora_state is None:  # pylint: disable=protected-access
    # Already inside a swap, or nothing to swap in: nothing to do.
    runner._frozen_lora_depth = depth + 1  # pylint: disable=protected-access
    try:
      yield
    finally:
      runner._frozen_lora_depth = depth  # pylint: disable=protected-access
    return

  live_lora_state = {
      name: param.data.detach().clone()
      for name, param in runner._backend.model.named_parameters()  # pylint: disable=protected-access
      if param.requires_grad
  }
  for name, param in runner._backend.model.named_parameters():  # pylint: disable=protected-access
    if name in runner._frozen_lora_state:  # pylint: disable=protected-access
      param.data.copy_(runner._frozen_lora_state[name])  # pylint: disable=protected-access

  runner._frozen_lora_depth = 1  # pylint: disable=protected-access
  try:
    yield
  finally:
    for name, param in runner._backend.model.named_parameters():  # pylint: disable=protected-access
      if name in live_lora_state:
        param.data.copy_(live_lora_state[name])
    runner._frozen_lora_depth = 0  # pylint: disable=protected-access


def _sample_policy_action(runner, state) -> int | None:
  """Sample one action from the policy for whoever is on move.

  Performs **no** weight swapping: the caller decides which weights are
  active, normally by wrapping the call in ``_frozen_lora_active``.

  Args:
    runner: The ``GRPORunner`` instance.
    state: The current game state (will NOT be modified).

  Returns:
    The selected action ID, or None if the state is terminal or has no
    legal actions.  Falls back to a uniformly random legal action when the
    completion cannot be parsed.
  """
  if state.is_terminal():
    return None
  current_player = state.current_player()
  legal = state.legal_actions(current_player)
  if not legal:
    return None

  # Render the state and legal actions for the player on move.
  state_text = runner._renderers[current_player].render_state(  # pylint: disable=protected-access
      state, current_player, runner._env.game  # pylint: disable=protected-access
  )
  legal_actions_with_desc = runner._renderers[  # pylint: disable=protected-access
      current_player
  ].render_legal_actions(state, current_player, runner._env.game)  # pylint: disable=protected-access
  legal_actions = [a for a, _ in legal_actions_with_desc]
  action_descriptions = [d for _, d in legal_actions_with_desc]

  prompt = runner._agents[current_player]._build_prompt(  # pylint: disable=protected-access
      state_text, legal_actions, action_descriptions
  )

  with torch.no_grad():
    response, _ = runner._backend.generate_with_logprobs(  # pylint: disable=protected-access
        prompt,
        temperature=runner._current_temperature,  # pylint: disable=protected-access
        max_tokens=runner._config.max_completion_length,  # pylint: disable=protected-access
    )

  action = runner._renderers[current_player].parse_action(  # pylint: disable=protected-access
      response, legal_actions_with_desc
  )
  if action is None:
    action = int(np.random.choice(legal_actions)) if legal_actions else 0
  return action


def _sample_llm_partner_action(runner, state) -> int | None:
  """Sample one action from the frozen LLM policy for the current player.

  Thin wrapper retained for callers that sample a *single* action outside
  a ``_frozen_lora_active`` block (``learn.action_reward``).  Anything
  sampling more than one action should open the context manager once and
  call ``_sample_policy_action`` in a loop.

  Args:
    runner: The ``GRPORunner`` instance.
    state: The current game state (will NOT be modified).

  Returns:
    The selected action ID, or None if not applicable.
  """
  with _frozen_lora_active(runner):
    return _sample_policy_action(runner, state)


def _rollout_policy_turns(runner, state, num_turns: int) -> int:
  """Play ``num_turns`` policy turns in place, alternating players.

  Extends the reward rollout's policy segment beyond the single candidate
  action.  With ``m = reward_policy_turns`` the caller has already applied
  the candidate action (turn 1), so this plays turns ``2 .. m``::

      p1_2, p0_3, ..., p0_m

  All turns are sampled from the frozen LoRA snapshot under a single weight
  swap.  Stops early on a terminal state or when no action can be produced.

  Args:
    runner: The ``GRPORunner`` instance.
    state: The game state, **modified in place**.
    num_turns: How many turns to play (``m - 1``).  Values <= 0 are no-ops.

  Returns:
    The number of turns actually played.
  """
  if num_turns <= 0:
    return 0

  played = 0
  with _frozen_lora_active(runner):
    for _ in range(num_turns):
      if state.is_terminal():
        break
      action = _sample_policy_action(runner, state)
      if action is None:
        break
      state.apply_action(action)
      played += 1
  return played


def _heuristic_rollout_score(
    runner, state, target_player: int, max_score: float = 25.0,
    seed: int | None = None, turn_discount: float = 1.0,
) -> float:
  """Roll out the game to terminal with heuristic play, return game score.

  Uses SafePlayPlayer if available, otherwise random legal actions.

  Args:
    runner: The ``GRPORunner`` instance.
    state: The current game state (will be modified in place).
    target_player: The player whose score to return.
    max_score: Maximum possible game score (25 for standard Hanabi).
    seed: Optional RNG seed for the heuristic partner.  ``SafePlayPlayer``
        picks a uniformly random legal hint when it has no known-playable
        card, so the rollout is stochastic.  Passing the *same* seed for
        every action in a GRPO group (common random numbers) makes that
        partner randomness common-mode, so it largely cancels in the
        within-group comparison that GRPO actually differentiates.
        ``None`` reproduces the previous behaviour (fresh randomness per
        call).
    turn_discount: Per-turn discount gamma applied to the terminal score,
        i.e. the return is ``gamma**turns * score``.  ``1.0`` disables it
        and reproduces the undiscounted behaviour.  See the note below.

  Returns:
    Normalized, turn-discounted game score in [0, 1].
  """
  # ── Why discount by turns ──
  # Without this, the reward is the terminal score of a SafePlayPlayer
  # continuation, which is dominated by the deal rather than by the
  # candidate action: every action in a GRPO group scores within a whisker
  # of every other, all rewards land in a narrow band around +0.1, and the
  # advantages are pure noise.  Worse, the only action class with any real
  # downside is *playing* (it can bomb), so the argmax of that reward is
  # "never play".  Runs 5454236 / 5454292 both collapsed into exactly that:
  # ~1% plays, games running to deck exhaustion at score 0.
  #
  # Discounting makes reaching a given score sooner strictly better, so a
  # successful play beats a hint twice over -- it raises the score and it
  # shortens the remaining game.
  #
  # It MUST be multiplicative, not an additive per-turn cost.  With an
  # additive cost, bombing out after 6 turns (score 0, small cost) scores
  # HIGHER than stalling 80 turns to a score of 2, so the reward would
  # actively teach the model to lose on purpose.  Multiplicatively, score 0
  # is a fixed point: gamma**6 * 0 == 0 < gamma**80 * 0.08.
  #
  # Honest caveat: Hanabi ends on deck exhaustion, and discards/plays draw
  # while hints do not, so discounting also mildly favours discarding over
  # hinting.  That is the intended direction here (the hint spiral is the
  # worse failure mode) but it is not a pure "progress" signal.
  #
  # Across-group level differences (late states have fewer turns left, so
  # larger gamma**turns) do not matter: TRL subtracts the group mean, and a
  # group is one prompt, hence one game state.
  try:
    from env.hanabi.heuristic_player import SafePlayPlayer  # pylint: disable=g-import-not-at-top

    heuristic = SafePlayPlayer(seed=seed)
  except ImportError:
    heuristic = None

  rng = np.random.RandomState(seed) if seed is not None else np.random

  game = getattr(runner._env, 'game', None)

  turns = 0
  while not state.is_terminal():
    player = state.current_player()
    legal = state.legal_actions(player)
    if not legal:
      break
    if heuristic is not None:
      action = heuristic.select_action(state, player, game)
      if action is None:
        action = int(rng.choice(legal))
    else:
      action = int(rng.choice(legal))
    state.apply_action(action)
    turns += 1

  # Extract terminal score.
  if state.is_terminal() and state.rewards() is not None:
    score = float(state.rewards()[target_player])
  elif hasattr(state, 'returns'):
    try:
      score = float(state.returns()[target_player])
    except (IndexError, TypeError):
      score = 0.0
  else:
    score = 0.0

  value = score / max_score
  if turn_discount != 1.0:
    value *= turn_discount**turns

  # Penalize non-scoring stalls: if the continuation finishes with score <= 1.0,
  # apply a stall penalty. This eliminates the zero fixed-point of multiplicative
  # turn discounting (where 0 * gamma^turns == 0) and creates a clear advantage gap
  # between actions that enable scoring vs passive hint/discard stalling.
  if score <= 1.0:
    value = -0.1 * (1.0 - score / 2.0)

  return value


def _rollout_value(
    runner,
    base_state,
    target_player: int,
    num_samples: int = 1,
    seed: int | None = None,
    max_score: float = 25.0,
    turn_discount: float = 1.0,
) -> float:
  """Estimate the value of ``base_state`` by averaging heuristic rollouts.

  ``base_state`` is the successor state -- the candidate action (and,
  optionally, one LLM partner response) has already been applied.  This
  function only estimates how good that state is.

  Because ``_heuristic_rollout_score`` mutates the state in place, each
  sample runs on a fresh clone.  Samples use consecutive seeds derived
  from ``seed``, so two different actions scored with the same ``seed``
  see the same sequence of partner RNG streams (common random numbers).

  Args:
    runner: The ``GRPORunner`` instance.
    base_state: The successor state to value (not modified).
    target_player: The player whose score to return.
    num_samples: Number of rollouts to average.  Rollouts cost
        milliseconds against a ~16-18 s LLM generation step, so modest
        values are effectively free.
    seed: Base seed for the rollouts.  ``None`` disables CRN.
    max_score: Maximum possible game score (25 for standard Hanabi).
    turn_discount: Per-turn discount applied to each rollout's terminal
        score; see ``_heuristic_rollout_score``.  ``1.0`` disables it.

  Returns:
    Mean normalized, turn-discounted game score in [0, 1].
  """
  num_samples = max(1, int(num_samples))

  # Fast path: single sample, no clone needed.
  if num_samples == 1:
    return _heuristic_rollout_score(
        runner,
        base_state,
        target_player,
        max_score=max_score,
        seed=seed,
        turn_discount=turn_discount,
    )

  scores = []
  for i in range(num_samples):
    sample_state = base_state.clone()
    scores.append(
        _heuristic_rollout_score(
            runner,
            sample_state,
            target_player,
            max_score=max_score,
            seed=None if seed is None else seed + i,
            turn_discount=turn_discount,
        )
    )
  return float(np.mean(scores))


def _group_rollout_seed(prompt_text: str, pass_idx: int) -> int:
  """Derive a stable per-group rollout seed for common random numbers.

  All completions in a GRPO group share one prompt (one game state), so
  hashing the prompt gives every candidate action in that group the same
  seed.  Mixing in ``pass_idx`` means the group is re-evaluated against a
  different partner trajectory on each pass, so the policy cannot overfit
  to one lucky rollout.

  ``zlib.crc32`` is used rather than ``hash()`` because the latter is
  salted per process (``PYTHONHASHSEED``) and would not be reproducible
  across runs.

  Args:
    prompt_text: The GRPO prompt (identifies the game state / group).
    pass_idx: The current GRPO pass index.

  Returns:
    A non-negative seed suitable for ``np.random.RandomState``.
  """
  base = zlib.crc32(prompt_text.encode('utf-8', errors='ignore'))
  # Knuth multiplicative mix so consecutive passes decorrelate.
  return int((base ^ (pass_idx * 2654435761)) & 0x7FFFFFFF)



def _get_target_token_ids(tokenizer, words: list[str]) -> list[int]:
  """Extract initial token IDs for a list of words from a tokenizer."""
  token_ids = set()
  for word in words:
    for variant in (
        word,
        ' ' + word,
        '\n' + word,
        word.lower(),
        ' ' + word.lower(),
    ):
      ids = tokenizer.encode(variant, add_special_tokens=False)
      if ids:
        token_ids.add(ids[0])
  return sorted(token_ids)


class ActionTypeLogitsProcessor:
  """Forces action-type diversity at the first generation token in GRPO groups.

  For a group of K completions generated for each prompt, a subset of
  completions have their first token restricted to specific action types
  (Play, Discard, Hint), while the remaining completions are generated
  freely from the model's unconstrained policy.

  This operates directly on token logits during generation, so the
  generated text and the evaluated action are 100% aligned.
  """

  def __init__(
      self,
      tokenizer,
      num_generations: int,
      forced_per_type: int = 2,
  ):
    self._tokenizer = tokenizer
    self._k = num_generations
    self._forced_per_type = forced_per_type

    # Cache token IDs for each action type's opening token.
    self._play_token_ids = _get_target_token_ids(tokenizer, ['Play'])
    self._discard_token_ids = _get_target_token_ids(tokenizer, ['Discard'])
    self._hint_token_ids = _get_target_token_ids(tokenizer, ['Hint'])

    # Track prompt length to identify token 1.
    self._prompt_len = None

  def __call__(
      self, input_ids: torch.LongTensor, scores: torch.FloatTensor
  ) -> torch.FloatTensor:
    cur_len = input_ids.shape[1]

    # Initialize or reset prompt length when a new generation batch starts.
    if self._prompt_len is None or cur_len < self._prompt_len:
      self._prompt_len = cur_len

    # Only modify logits at the very first generated token (t = 1).
    if cur_len != self._prompt_len:
      return scores

    batch_size = scores.shape[0]
    n_forced = self._forced_per_type

    # Slots within each group of K:
    # [0 .. K - 3*n_forced - 1]: free policy
    # [K - 3*n_forced .. K - 2*n_forced - 1]: forced Play
    # [K - 2*n_forced .. K - n_forced - 1]: forced Discard
    # [K - n_forced .. K - 1]: forced Hint
    play_start = max(0, self._k - 3 * n_forced)
    discard_start = max(0, self._k - 2 * n_forced)
    hint_start = max(0, self._k - n_forced)

    neg_inf = -float('inf')
    for b in range(batch_size):
      group_idx = b % self._k
      target_ids = None
      if play_start <= group_idx < discard_start and self._play_token_ids:
        target_ids = self._play_token_ids
      elif discard_start <= group_idx < hint_start and self._discard_token_ids:
        target_ids = self._discard_token_ids
      elif hint_start <= group_idx < self._k and self._hint_token_ids:
        target_ids = self._hint_token_ids

      if target_ids is not None:
        mask = torch.full_like(scores[b], neg_inf)
        for tid in target_ids:
          mask[tid] = scores[b, tid]
        scores[b] = mask

    return scores


class StrategicActionLogitsProcessor:
  """Forces strategic actions into GRPO completion groups during generation.

  For each prompt in the batch (replicated K times), analyzes the game state
  to select high-value strategic actions across priority tiers:
    1. Known-safe plays (card fully hinted and playable)
    2. Risky plays (partially hinted, good playability potential)
    3. Smart discards (known dead or oldest unhinted)
    4. Diverse hints (touching playable partner cards, diverse types)

  For each selected action, forces the full token sequence token-by-token
  into one completion slot within the group of K, followed by EOS.
  The remaining slots are left for unconstrained policy generation.
  """

  def __init__(
      self,
      tokenizer,
      runner,
      num_generations: int,
      max_forced_fraction: float = 0.5,
  ):
    self._tokenizer = tokenizer
    self._runner = runner
    self._k = num_generations
    self._max_forced_fraction = max_forced_fraction

    # Track prompt length to identify the start of generation (step 0).
    self._prompt_len = None
    # Map from batch_index (0 .. batch_size - 1) -> list of forced token IDs.
    self._forced_tokens_by_batch: dict[int, list[int]] = {}
    self._eos_token_id = tokenizer.eos_token_id
    self._pad_token_id = (
        getattr(tokenizer, 'pad_token_id', None) or self._eos_token_id
    )

  def _prepare_forced_tokens(
      self, input_ids: torch.LongTensor
  ) -> dict[int, list[int]]:
    """Analyze game states for prompts in the batch and determine forced tokens."""
    forced_map: dict[int, list[int]] = {}
    batch_size = input_ids.shape[0]
    num_groups = max(1, batch_size // self._k)

    for g in range(num_groups):
      group_start = g * self._k
      if group_start >= batch_size:
        break

      # Decode prompt text to look up metadata.
      prompt_tokens = input_ids[group_start, : self._prompt_len]
      prompt_text = self._tokenizer.decode(
          prompt_tokens, skip_special_tokens=True
      ).strip()

      metadata = self._runner._prompt_metadata.get(prompt_text, None)
      if metadata is None:
        # Fallback: search prompt_metadata by prefix or substring.
        for k, v in self._runner._prompt_metadata.items():
          if k.strip() == prompt_text or prompt_text in k or k in prompt_text:
            metadata = v
            break

      if not metadata or 'serialized_state' not in metadata:
        continue

      ser_state = metadata['serialized_state']
      p_id = metadata.get('player_id', 0)
      legal_actions_desc = metadata.get('legal_actions_desc', [])
      if not legal_actions_desc:
        continue

      try:
        from env.hanabi.hanabi_env import deserialize_game_and_state  # pylint: disable=g-import-not-at-top
        _, state = deserialize_game_and_state(ser_state)
      except (ImportError, Exception):
        _, state = _deserialize_game_and_state(ser_state)

      try:
        from learn.strategic_actions import analyze_strategic_actions  # pylint: disable=g-import-not-at-top
        plan = analyze_strategic_actions(
            state=state,
            player_id=p_id,
            legal_actions_desc=legal_actions_desc,
            num_generations=self._k,
            max_forced_fraction=self._max_forced_fraction,
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning('Failed to analyze strategic actions: %s', e)
        continue

      if not plan.forced_action_texts:
        continue

      # Assign forced actions to the last M slots of this prompt's group.
      m_forced = len(plan.forced_action_texts)
      for m_idx, action_text in enumerate(plan.forced_action_texts):
        slot = self._k - m_forced + m_idx
        b_idx = group_start + slot
        if b_idx >= batch_size:
          break

        # Tokenize with leading space first (standard continuation after
        # prompt colon).
        tokens = self._tokenizer.encode(
            ' ' + action_text, add_special_tokens=False
        )
        if not tokens:
          tokens = self._tokenizer.encode(action_text, add_special_tokens=False)
        if tokens:
          forced_map[b_idx] = tokens

    return forced_map

  def __call__(
      self, input_ids: torch.LongTensor, scores: torch.FloatTensor
  ) -> torch.FloatTensor:
    cur_len = input_ids.shape[1]

    # Initialize on the first generation token (or reset if batch restarted).
    if self._prompt_len is None or cur_len < self._prompt_len:
      self._prompt_len = cur_len
      self._forced_tokens_by_batch = self._prepare_forced_tokens(input_ids)

    step = cur_len - self._prompt_len
    if step < 0 or not self._forced_tokens_by_batch:
      return scores

    neg_inf = -float('inf')
    for b, target_token_ids in self._forced_tokens_by_batch.items():
      if b >= scores.shape[0]:
        continue

      if step < len(target_token_ids):
        # Force the next token in the sequence.
        target_tid = target_token_ids[step]
        scores[b, :] = neg_inf
        scores[b, target_tid] = 0.0
      else:
        # Action string complete: force EOS (or PAD) to terminate generation.
        term_id = (
            self._eos_token_id
            if self._eos_token_id is not None
            else self._pad_token_id
        )
        if term_id is not None:
          scores[b, :] = neg_inf
          scores[b, term_id] = 0.0

    return scores


def _lookup_prompt_metadata(runner, prompt_text: str):
  """Finds the metadata entry for a decoded prompt.

  Decoded prompts do not always round-trip byte-for-byte (chat templates
  and special-token stripping both perturb them), so fall back to a
  containment match before giving up.

  Args:
    runner: The ``GRPORunner`` holding ``_prompt_metadata``.
    prompt_text: Decoded prompt text.

  Returns:
    The metadata dict, or ``None`` when nothing matches.
  """
  metadata = runner._prompt_metadata.get(prompt_text)  # pylint: disable=protected-access
  if metadata is not None:
    return metadata
  stripped = prompt_text.strip()
  for key, value in runner._prompt_metadata.items():  # pylint: disable=protected-access
    if key.strip() == stripped or stripped in key or key in stripped:
      return value
  return None


class GroupDiversifier:
  """De-duplicates a GRPO completion group and back-fills strategic actions.

  ``StrategicActionLogitsProcessor`` forces actions into fixed slots
  *before* generation, so it cannot know what the policy was about to
  produce and routinely spends a slot on an action the model would have
  sampled anyway.  Measured on a live run, a group of K=16 completions
  collapsed to roughly 3 distinct actions (the ``Reward cache`` log line
  reported ~80% duplicates), which leaves GRPO comparing an action almost
  entirely against itself.

  This class runs *after* generation instead:

    1. Decode all K completions and parse each to an action ID with the
       same renderer ``reward_fn`` uses, so the de-duplication key is
       exactly the key rewards are cached under.
    2. Keep the first completion for each distinct action.  Every later
       repeat -- and every completion that fails to parse, since those
       are assigned a uniformly random legal action downstream -- frees
       its slot.
    3. Fill freed slots with actions not yet represented in the group,
       best tier first (known-safe plays, risky plays, smart discards,
       diverse hints), then any remaining legal action as a floor.
    4. Overwrite those slots' completion token IDs in place.

  Because the substitution rewrites the tokens themselves, the text TRL
  backpropagates through and the action the reward function scores stay
  aligned.  This is the mismatch that ``reward_fn`` warns about and that
  rules out substituting action IDs at reward time.

  Substituted completions are off-policy -- the policy did not propose
  them.  TRL recomputes log-probabilities from the final token IDs, so
  the gradient is well formed, but it is REINFORCE on forced samples
  rather than on-policy GRPO.  That is the intent: it is what makes the
  group span distinct actions early in training, when the policy has not
  yet learnt that anything other than a hint exists.
  """

  def __init__(
      self,
      tokenizer,
      runner,
      num_generations: int,
      max_substitution_fraction: float = 1.0,
  ):
    self._tokenizer = tokenizer
    self._runner = runner
    self._k = num_generations
    self._max_fraction = max_substitution_fraction
    self._eos_token_id = tokenizer.eos_token_id
    pad_id = getattr(tokenizer, 'pad_token_id', None)
    self._pad_token_id = pad_id if pad_id is not None else tokenizer.eos_token_id
    # Number of groups seen; used to throttle the per-action detail block.
    self._group_counter = 0
    # Pass-level totals, reported by ``log_summary``.
    self.total_groups = 0
    self.total_sampled_distinct = 0
    self.total_final_distinct = 0
    self.total_substituted = 0
    self.total_slots_left_duplicate = 0

  def _encode_completion(self, text: str, width: int, device) -> torch.Tensor:
    """Tokenises a substituted action to exactly ``width`` token IDs.

    Args:
      text: The action description to write into the slot.
      width: Completion width of the generated tensor.
      device: Device of the tensor being written.

    Returns:
      A 1-D tensor of length ``width``: the action tokens, EOS, then pad.
    """
    # The prompt ends with "Action:\n", so completions begin with a
    # leading space in the model's own samples; match that.
    ids = self._tokenizer.encode(' ' + text, add_special_tokens=False)
    if not ids:
      ids = self._tokenizer.encode(text, add_special_tokens=False)
    if self._eos_token_id is not None:
      # EOS must survive truncation: TRL builds the completion mask from
      # the first EOS, and a slot with no EOS is masked to full width.
      if len(ids) >= width:
        logging.warning(
            '[diversify] completion %r has %d tokens >= width %d; truncating',
            text,
            len(ids),
            width,
        )
        ids = ids[: max(width - 1, 0)]
      ids = ids + [self._eos_token_id]
    ids = ids[:width]
    pad = self._pad_token_id if self._pad_token_id is not None else 0
    ids = ids + [pad] * (width - len(ids))
    return torch.tensor(ids, dtype=torch.long, device=device)

  def _candidate_actions(
      self,
      ser_state,
      player_id: int,
      legal_actions_desc: list[tuple[int, str]],
      present: set[int],
  ) -> list[tuple[int, str, str]]:
    """Ranks actions missing from the group, best first.

    Args:
      ser_state: Serialized game+state for this prompt.
      player_id: The acting player.
      legal_actions_desc: ``(action_id, description)`` for every legal action.
      present: Action IDs already covered by a kept completion.

    Returns:
      ``(action_id, description, tier)`` triples, highest value first.
    """
    text_by_id = dict(legal_actions_desc)
    ordered: list[tuple[int, str, str]] = []
    chosen: set[int] = set()

    try:
      from learn.strategic_actions import analyze_strategic_actions  # pylint: disable=g-import-not-at-top

      _, state = _deserialize_game_and_state(ser_state)
      plan = analyze_strategic_actions(
          state=state,
          player_id=player_id,
          legal_actions_desc=legal_actions_desc,
          num_generations=self._k,
          # Rank every tier; the caller caps how many are actually used.
          max_forced_fraction=1.0,
      )
      tiers = (
          ('safe_play', plan.known_safe_plays),
          ('risky_play', plan.risky_plays),
          ('smart_discard', plan.smart_discards),
          ('diverse_hint', plan.diverse_hints),
      )
      for tier_name, action_ids in tiers:
        for aid in action_ids:
          if aid in present or aid in chosen:
            continue
          desc = plan.action_texts.get(aid) or text_by_id.get(aid, '')
          if desc:
            ordered.append((aid, desc, tier_name))
            chosen.add(aid)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning('[diversify] strategic analysis failed: %s', e)

    # Diversity floor: the strategic tiers can come back short (e.g. no
    # safe play exists and every hint duplicates one already sampled).
    # An unexplored legal action is still worth more than a duplicate.
    for aid, desc in legal_actions_desc:
      if aid in present or aid in chosen or not desc:
        continue
      ordered.append((aid, desc, 'legal_fill'))
      chosen.add(aid)

    return ordered

  def _diversify_group(
      self,
      sequences: torch.Tensor,
      prompt_len: int,
      width: int,
      group_start: int,
      max_subs: int,
  ) -> None:
    """De-duplicates and back-fills one group of K completions in place."""
    tokenizer = self._tokenizer
    prompt_text = tokenizer.decode(
        sequences[group_start, :prompt_len], skip_special_tokens=True
    ).strip()
    metadata = _lookup_prompt_metadata(self._runner, prompt_text)
    if not metadata:
      return
    legal_actions_desc = metadata.get('legal_actions_desc') or []
    ser_state = metadata.get('serialized_state')
    if not legal_actions_desc or ser_state is None:
      return
    player_id = metadata.get('player_id', 0)
    renderer = self._runner._renderers[player_id]  # pylint: disable=protected-access

    texts: list[str] = []
    action_ids: list[int | None] = []
    for j in range(self._k):
      text = tokenizer.decode(
          sequences[group_start + j, prompt_len:], skip_special_tokens=True
      ).strip()
      texts.append(text)
      action_ids.append(renderer.parse_action(text, legal_actions_desc))

    # First completion per distinct action keeps its slot; repeats and
    # parse failures free theirs.
    kept: dict[int, int] = {}
    counts: dict[int, int] = {}
    free_slots: list[int] = []
    num_unparsed = 0
    for j, aid in enumerate(action_ids):
      if aid is None:
        num_unparsed += 1
        free_slots.append(j)
        continue
      counts[aid] = counts.get(aid, 0) + 1
      if aid in kept:
        free_slots.append(j)
      else:
        kept[aid] = j

    candidates = self._candidate_actions(
        ser_state, player_id, legal_actions_desc, set(kept)
    )
    num_subs = min(len(free_slots), len(candidates), max_subs)

    substitutions: list[tuple[int, int, str, str]] = []
    device = sequences.device
    is_reasoning = getattr(self._runner._config, 'reasoning', False)
    cot_bot = None
    cot_state = None
    cot_fn = None
    if is_reasoning:
      try:
        from env.hanabi.heuristic_player import SafePlayPlayer  # pylint: disable=g-import-not-at-top
        from data.generate_bc_data import generate_cot_reasoning  # pylint: disable=g-import-not-at-top
        _, cot_state = _deserialize_game_and_state(ser_state)
        cot_bot = SafePlayPlayer(seed=42)
        cot_fn = generate_cot_reasoning
      except Exception:  # pylint: disable=broad-exception-caught
        cot_bot = None
        cot_state = None
        cot_fn = None

    for slot, (aid, desc, tier) in zip(free_slots[:num_subs], candidates):
      completion_text = desc
      if is_reasoning and cot_bot is not None and cot_state is not None and cot_fn is not None:
        try:
          completion_text = cot_fn(cot_state, player_id, desc, cot_bot)
        except Exception:  # pylint: disable=broad-exception-caught
          completion_text = desc
      sequences[group_start + slot, prompt_len:] = self._encode_completion(
          completion_text, width, device
      )
      substitutions.append((slot, aid, desc, tier))

    self._group_counter += 1
    self.total_groups += 1
    self.total_sampled_distinct += len(kept)
    self.total_final_distinct += len(kept) + num_subs
    self.total_substituted += num_subs
    self.total_slots_left_duplicate += len(free_slots) - num_subs

    logging.info(
        '[diversify] P%d group: K=%d | model sampled %d distinct '
        '(%d unparseable) -> %d distinct after %d substitutions '
        '| %d slots still duplicate',
        player_id,
        self._k,
        len(kept),
        num_unparsed,
        len(kept) + num_subs,
        num_subs,
        len(free_slots) - num_subs,
    )

    # Per-action detail for the first few groups of each pass: enough to
    # see the collapse and the fix without flooding the log.
    if self._group_counter <= 5:
      for aid, slot in sorted(
          kept.items(), key=lambda kv: -counts.get(kv[0], 0)
      ):
        logging.info(
            '[diversify]   kept  x%-2d a=%-3d %r',
            counts.get(aid, 1),
            aid,
            texts[slot],
        )
      if num_unparsed:
        logging.info(
            '[diversify]   unparseable x%d (slots freed)', num_unparsed
        )
      for slot, aid, desc, tier in substitutions:
        logging.info(
            '[diversify]   +sub  slot %-2d a=%-3d [%s] %r',
            slot,
            aid,
            tier,
            desc,
        )

  def diversify(self, sequences: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Rewrites duplicate completion slots across every group in a batch.

    Args:
      sequences: ``[batch, prompt_len + completion_len]`` generated IDs.
        Modified in place where possible, or reallocated if width expands.
      prompt_len: Number of prompt tokens prefixed to every row.

    Returns:
      ``sequences`` (possibly extended along sequence dimension).
    """
    if not torch.is_tensor(sequences) or sequences.dim() != 2:
      return sequences
    total, seq_len = sequences.shape
    width = seq_len - prompt_len
    if width <= 0 or self._k <= 1 or total < self._k:
      return sequences
    max_subs = max(0, int(self._k * self._max_fraction))
    if max_subs == 0:
      return sequences

    # Expand width if current generated completion length is shorter than
    # max_completion_length, so that longer substituted actions (e.g. rank hints)
    # are never truncated.
    target_width = max(
        width, getattr(self._runner._config, 'max_completion_length', 20)
    )
    if target_width > width:
      pad_id = (
          self._pad_token_id
          if self._pad_token_id is not None
          else (self._eos_token_id or 0)
      )
      pad_len = target_width - width
      padding = torch.full(
          (total, pad_len),
          pad_id,
          dtype=sequences.dtype,
          device=sequences.device,
      )
      sequences = torch.cat([sequences, padding], dim=1)
      width = target_width

    for group_start in range(0, total - self._k + 1, self._k):
      try:
        self._diversify_group(
            sequences, prompt_len, width, group_start, max_subs
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        # A failure here must never take down training: the group simply
        # stays as the model sampled it.
        logging.warning(
            '[diversify] group at batch index %d skipped: %s', group_start, e
        )
    return sequences

  def log_summary(self, pass_idx: int, player_id) -> None:
    """Logs pass-level de-duplication totals."""
    if not self.total_groups:
      logging.warning(
          '[diversify] pass %d: NO groups were processed -- the generate '
          'wrapper never fired or no prompt metadata matched.',
          pass_idx,
      )
      return
    logging.info(
        '[diversify] pass %d P%s SUMMARY: %d groups | mean distinct '
        '%.2f -> %.2f of K=%d | %d substitutions | %d slots left duplicate',
        pass_idx,
        player_id,
        self.total_groups,
        self.total_sampled_distinct / self.total_groups,
        self.total_final_distinct / self.total_groups,
        self._k,
        self.total_substituted,
        self.total_slots_left_duplicate,
    )


# ═══════════════════════════════════════════════════════════════════════
# TRL-based GRPO training step
# ═══════════════════════════════════════════════════════════════════════


def _build_prompt_dataset(prompts: list[str]):
  """Build a HuggingFace Dataset from prompt strings."""
  from datasets import Dataset  # pylint: disable=g-import-not-at-top

  return Dataset.from_dict({'prompt': prompts})


def _resolve_scale_rewards(trl_module, scale_mode: str) -> dict:
  """Resolve the ``scale_rewards`` kwarg for the installed TRL version.

  TRL always subtracts the group mean when forming advantages; this setting
  picks the *divisor*.  Left unset it defaults to the within-group std,
  which rescales a group of near-identical rewards (std ~ 1e-3, i.e. pure
  rollout noise) to the same +/-1 advantages as a group with real spread.
  ``'batch'`` divides by the batch-wide std instead, so advantage magnitude
  tracks the size of the actual reward difference.

  The accepted type changed across TRL releases: newer versions take the
  strings ``'group'`` / ``'batch'`` / ``'none'``, older ones take a bool.
  Probe the dataclass field rather than crashing on an unknown value.

  Args:
    trl_module: The imported ``trl`` module.
    scale_mode: One of ``'group'``, ``'batch'``, ``'none'``.

  Returns:
    A kwargs dict to splat into ``GRPOConfig``.  Empty when the installed
    TRL exposes no ``scale_rewards`` field, leaving the library default.
  """
  try:
    fields = {f.name: f for f in dataclasses.fields(trl_module.GRPOConfig)}
  except (TypeError, AttributeError) as e:
    logging.warning(
        'Could not probe TRL scale_rewards (%s); leaving library default.', e
    )
    return {}

  if 'scale_rewards' not in fields:
    logging.warning(
        'Installed TRL GRPOConfig has no scale_rewards field; advantage '
        'scaling left at the library default.'
    )
    return {}

  annotation = str(fields['scale_rewards'].type)
  if 'bool' in annotation and 'str' not in annotation:
    # Legacy bool API: True divides by the group std, False does not.
    if scale_mode == 'batch':
      logging.warning(
          "Installed TRL types scale_rewards as bool; 'batch' is "
          'unavailable, falling back to group scaling.'
      )
    return {'scale_rewards': scale_mode != 'none'}

  return {'scale_rewards': scale_mode}


def _cleanup_ref_adapter(model, active_adapter: str | None = None) -> None:
  """Removes the temporary 'ref' adapter created by TRL's GRPOTrainer.

  TRL's GRPOTrainer automatically adds a frozen reference adapter named 'ref'
  to a PeftModel when beta > 0. Because we train iteratively across players
  and passes with the same backend model instance, subsequent GRPOTrainer
  initializations will fail with:
      ValueError: Adapter with name 'ref' already exists.
  This helper cleans up the 'ref' adapter and restores the active adapter.
  """
  actual_model = getattr(model, 'module', model)
  if hasattr(actual_model, 'delete_adapter'):
    peft_config = getattr(actual_model, 'peft_config', None)
    if peft_config and 'ref' in peft_config:
      try:
        actual_model.delete_adapter('ref')
        logging.info("Cleaned up temporary 'ref' adapter from model.")
      except Exception as e:
        logging.warning("Failed to delete 'ref' adapter: %s", e)
  if active_adapter and hasattr(actual_model, 'set_adapter'):
    try:
      peft_config = getattr(actual_model, 'peft_config', None)
      if peft_config and active_adapter in peft_config:
        actual_model.set_adapter(active_adapter)
    except Exception as e:
      logging.warning("Failed to restore active adapter '%s': %s", active_adapter, e)


def _train_grpo_on_prompts(
    runner,
    unique_prompts: list[str],
    pass_idx: int,
    player_id: int | None,
    trl_module,
) -> tuple[float, float]:
  """Run one GRPO training step on a set of prompts via TRL.

  Args:
    runner: The ``GRPORunner`` instance.
    unique_prompts: Deduplicated prompt strings.
    pass_idx: Current pass index.
    player_id: Player-specific update (``None`` for combined).
    trl_module: The imported ``trl`` module.

  Returns:
    Tuple of ``(mean_loss, mean_reward)``.
  """
  eval_counter = [0]
  # Monotonic GRPO-group id within this pass.  Lives outside reward_fn
  # because TRL calls reward_fn once per generation batch and we want
  # group numbers that keep counting up across the whole pass.
  group_counter = [0]

  def reward_fn(completions, prompts=None, **kwargs):
    del kwargs
    rewards = []
    reward_cache = {}  # (prompt_text, action_id) -> reward tensor
    group_records = []  # one dict per completion, for the group/z-score dump
    _action_type_tracker = {}  # prompt_text -> {play, discard, hint, total}
    cache_hits = 0
    for i, completion in enumerate(completions):
      prompt_text = (
          prompts[i] if prompts is not None and i < len(prompts) else ''
      )
      metadata = runner._prompt_metadata.get(prompt_text, {})
      action_history = metadata.get('action_history', [])
      p_id = metadata.get('player_id', 0)
      legal_actions = metadata.get('legal_actions', [])
      legal_actions_desc = metadata.get('legal_actions_desc', [])
      ser_state = metadata.get('serialized_state', None)

      if hasattr(completion, 'text'):
        comp_text = completion.text
      elif isinstance(completion, list):
        comp_text = runner._backend.tokenizer.decode(
            completion, skip_special_tokens=True
        )
      else:
        comp_text = str(completion)

      action_id = runner._renderers[p_id].parse_action(
          comp_text, legal_actions_desc
      )
      parsed = True
      if action_id is None:
        parsed = False
        action_id = int(np.random.choice(legal_actions)) if legal_actions else 0

      # If parsing failed, apply parse failure penalty immediately without
      # touching the reward cache. This prevents parse failures from poisoning
      # legitimate actions in the cache, and prevents unparseable text from
      # receiving cached positive rewards.
      if not parsed:
        parse_penalty = -0.3
        reward_tensor = torch.tensor(float(parse_penalty))
        rewards.append(reward_tensor)
        eval_counter[0] += 1
        group_records.append({
            'prompt': prompt_text,
            'player': p_id,
            'text': comp_text.strip(),
            'action': action_id,
            'status': 'random_fallback',
            'reward': float(parse_penalty),
        })
        if eval_counter[0] <= 5 or eval_counter[0] % 25 == 0:
          logging.info(
              '[GRPO eval #%d] P%d | completion=%r '
              '-> action=%s (random_fallback) | reward=%.1f',
              eval_counter[0],
              p_id,
              comp_text.strip(),
              action_id,
              parse_penalty,
          )
        continue

      # Check reward cache for duplicate (prompt, action) pairs (parsed only).
      cache_key = (prompt_text, action_id)
      if cache_key in reward_cache:
        rewards.append(reward_cache[cache_key])
        cache_hits += 1
        eval_counter[0] += 1
        group_records.append({
            'prompt': prompt_text,
            'player': p_id,
            'text': comp_text.strip(),
            'action': action_id,
            'status': 'parsed/cached',
            'reward': float(reward_cache[cache_key]),
        })
        if eval_counter[0] <= 5 or eval_counter[0] % 25 == 0:
          logging.info(
              '[GRPO eval #%d] P%d | completion=%r '
              '-> action=%s (parsed) | reward=%.1f (cached)',
              eval_counter[0],
              p_id,
              comp_text.strip(),
              action_id,
              float(reward_cache[cache_key]),
          )
        continue

      # ── Constrained action type diversity (diagnostic) ──
      # Track action types per prompt group.  When enabled, log group
      # diversity stats so we can measure collapse.  Actual constrained
      # generation (prefix-forcing) must happen at the TRL generation
      # level, NOT by substituting action_ids in the reward function,
      # because substitution creates a mismatch between the generated
      # text (which TRL backprops through) and the scored action.
      if (
          (
              runner._config.constrained_action_types
              or runner._config.strategic_action_selection
          )
          and ser_state is not None
          and legal_actions
      ):
        if prompt_text not in _action_type_tracker:
          _action_type_tracker[prompt_text] = {
              'play': 0,
              'discard': 0,
              'hint': 0,
              'total': 0,
          }
        tracker = _action_type_tracker[prompt_text]
        tracker['total'] += 1

        _, cls_state = _deserialize_game_and_state(ser_state)
        runner._env.set_state(cls_state)
        action_type = _classify_hanabi_action_type(cls_state, action_id, p_id)
        tracker[action_type] += 1

        k = runner._config.num_generations
        if tracker['total'] == k:
          # Log group diversity at the end of each prompt group.
          logging.info(
              '[action_diversity] P%d group complete: '
              'play=%d discard=%d hint=%d / K=%d',
              p_id,
              tracker['play'],
              tracker['discard'],
              tracker['hint'],
              k,
          )

      sim_mode = runner._config.reward_simulation_mode

      if sim_mode == 'dense':
        # Dense per-action reward: evaluate the immediate quality of
        # the chosen action without any forward simulation.
        from learn.action_reward import evaluate_action_quality  # pylint: disable=g-import-not-at-top

        if not parsed:
          reward = -0.3  # Parse failure penalty.
        else:
          # Restore the state to evaluate the action.
          if ser_state is not None:
            _, eval_state = _deserialize_game_and_state(ser_state)
          else:
            runner._env.reset()
            eval_state = runner._env._state
            for a in action_history:
              if eval_state.is_terminal():
                break
              eval_state.apply_action(a)
          reward = evaluate_action_quality(eval_state, action_id, p_id)

      elif sim_mode == 'dense_chain':
        # Dense rewards over a short heuristic continuation,
        # optionally blended with a heuristic rollout game score
        # and/or using an LLM partner response.
        from learn.action_reward import evaluate_dense_chain  # pylint: disable=g-import-not-at-top

        if not parsed:
          reward = -0.3  # Parse failure penalty.
        else:
          blend_w = runner._config.reward_blend_weight

          # ── Dense per-action chain term ──
          # At blend_w >= 1.0 the dense term is multiplied by zero, so
          # skip it entirely: evaluate_dense_chain runs a horizon-length
          # heuristic continuation and is the expensive half of this
          # branch.  This makes w=1.0 a genuine pure-rollout objective.
          if blend_w >= 1.0:
            primary_reward = 0.0
          else:
            horizon = runner._config.truncated_rollout_horizon or 4
            discount = getattr(runner._config, 'dense_chain_discount', 0.9)
            primary_reward = evaluate_dense_chain(
                runner,
                action_history,
                action_id,
                p_id,
                serialized_state=ser_state,
                horizon=horizon,
                discount=discount,
                llm_partner_response=(runner._config.llm_partner_response),
            )

          # ── Blend with heuristic rollout game score ──
          if blend_w > 0:
            # Restore state, apply action, optionally get LLM partner
            # response, then heuristic rollout to terminal.
            if ser_state is not None:
              _, blend_state = _deserialize_game_and_state(ser_state)
              runner._env.set_state(blend_state)
            else:
              runner._env.reset()
              blend_state = runner._env._state
              for a in action_history:
                if blend_state.is_terminal():
                  break
                blend_state.apply_action(a)
              blend_state = runner._env._state

            if not blend_state.is_terminal():
              blend_state.apply_action(action_id)

            # ── Policy continuation ──
            # The candidate action is turn 1 of the policy segment; play
            # turns 2..m from the frozen policy, alternating players, then
            # hand over to the heuristic.  Odd m ends the segment on the
            # acting player, so it has seen and responded to one partner
            # move -- the shortest rollout in which a hint can actually pay
            # off.  The legacy llm_partner_response flag is exactly m=2.
            policy_turns = runner._config.reward_policy_turns
            if runner._config.llm_partner_response and policy_turns < 2:
              policy_turns = 2
            if not blend_state.is_terminal():
              _rollout_policy_turns(runner, blend_state, policy_turns - 1)

            # Lives remaining after the whole policy segment.  Capturing
            # them here rather than immediately after the candidate action
            # is deliberate: life tokens only ever decrease, so this is the
            # minimum over the segment, and every move in the segment is
            # the policy's own.  Capturing earlier would make a bomb at
            # turn 3 invisible, reinstating exactly the blindness this
            # factor exists to remove.  At m=1 the two are identical.
            survival_exp = runner._config.reward_survival_exponent
            lives_after = None
            if survival_exp > 0 and hasattr(blend_state, 'life_tokens'):
              lives_after = blend_state.life_tokens()

            # Common random numbers: every completion in this group
            # shares one prompt, hence one seed, so the heuristic
            # partner's coin flips are common-mode across the candidate
            # actions GRPO compares against each other.
            group_seed = (
                _group_rollout_seed(prompt_text, pass_idx)
                if runner._config.reward_rollout_common_seed
                else None
            )
            game_score_norm = _rollout_value(
                runner,
                blend_state,
                p_id,
                num_samples=runner._config.reward_rollout_samples,
                seed=group_seed,
                turn_discount=runner._config.reward_turn_discount,
            )

            # ── Convex survival factor ──
            # SafePlayPlayer never loses a life (it plays only cards it
            # KNOWS are playable), so the rollout score is provably
            # invariant to lives remaining: 3->2 and 2->1 both move it by
            # exactly 0.0000.  The reward is therefore blind to the first
            # two bombs and dumps the whole penalty on the third.  This
            # restores the missing gradient.
            if lives_after is not None:
              max_lives = 3
              game_obj = getattr(runner._env, 'game', None)
              params = getattr(game_obj, '_params', None)
              if isinstance(params, dict):
                max_lives = params.get('max_life_tokens', 3)
              life_fraction = max(lives_after, 0) / max(max_lives, 1)
              if game_score_norm >= 0:
                game_score_norm *= life_fraction ** survival_exp
              else:
                # When negative (stall penalty), losing lives increases the penalty monotonically
                game_score_norm -= (1.0 - life_fraction) * 0.3

            reward = (1 - blend_w) * primary_reward + blend_w * game_score_norm
          else:
            reward = primary_reward

      elif (
          runner._config.reward_num_simulations > 1
          and runner._config.reward_variance_penalty > 0
      ):
        sim_rewards = [
            simulate_from_state(
                runner, action_history, action_id, p_id, ser_state
            )
            for _ in range(runner._config.reward_num_simulations)
        ]
        mean_r = float(np.mean(sim_rewards))
        std_r = float(np.std(sim_rewards))
        reward = mean_r - runner._config.reward_variance_penalty * std_r
      else:
        reward = simulate_from_state(
            runner, action_history, action_id, p_id, ser_state
        )

      reward_tensor = torch.tensor(float(reward))
      rewards.append(reward_tensor)
      reward_cache[cache_key] = reward_tensor

      eval_counter[0] += 1
      status = 'parsed'
      group_records.append({
          'prompt': prompt_text,
          'player': p_id,
          'text': comp_text.strip(),
          'action': action_id,
          'status': status,
          'reward': float(reward),
      })
      if eval_counter[0] <= 5 or eval_counter[0] % 25 == 0:
        logging.info(
            '[GRPO eval #%d] P%d | completion=%r '
            '-> action=%s (parsed) | reward=%.1f',
            eval_counter[0],
            p_id,
            comp_text.strip(),
            action_id,
            reward,
        )

    if cache_hits > 0:
      logging.info(
          'Reward cache: %d/%d hits (%.0f%% duplicates avoided)',
          cache_hits,
          len(rewards),
          100.0 * cache_hits / len(rewards),
      )

    # ── Per-group reward / z-score dump ──
    # One GRPO group == one prompt == one game state, and the K completions
    # in it are K candidate actions.  TRL ALWAYS subtracts the group mean;
    # only the denominator depends on ``grpo_scale_rewards`` ('group' ->
    # group std, 'batch' -> batch std, 'none' -> 1).  The z-score logged
    # here is the 'group' normalisation, (r - mean) / (std + 1e-4), which
    # is the exact advantage under scale_rewards='group' and is
    # proportional to it under 'batch'.
    #
    # The point of this dump is diagnostic: a group whose rewards are all
    # equal has std=0, every advantage is 0, and the group contributes NO
    # gradient no matter how many tokens it cost to generate.  Counting
    # those lines tells us directly how much of a pass was wasted.
    groups = {}
    for rec in group_records:
      groups.setdefault(rec['prompt'], []).append(rec)
    for recs in groups.values():
      group_counter[0] += 1
      vals = np.asarray([r['reward'] for r in recs], dtype=np.float64)
      mean_r = float(vals.mean())
      std_r = float(vals.std())
      gid = group_counter[0]
      logging.info(
          '[GRPO group #%d] P%d | n=%d distinct_actions=%d | '
          'reward mean=%+.4f std=%.4f min=%+.4f max=%+.4f%s',
          gid,
          recs[0]['player'],
          len(recs),
          len({r['action'] for r in recs}),
          mean_r,
          std_r,
          float(vals.min()),
          float(vals.max()),
          '  *** DEGENERATE: std=0, zero gradient ***'
          if std_r < 1e-6
          else '',
      )
      for slot, rec in enumerate(recs):
        adv = rec['reward'] - mean_r
        logging.info(
            '[GRPO group #%d]   [%d] a=%-3s %-15s r=%+.4f adv=%+.4f '
            'z=%+.4f | %r',
            gid,
            slot,
            rec['action'],
            rec['status'],
            rec['reward'],
            adv,
            adv / (std_r + 1e-4),
            rec['text'],
        )

    return rewards

  # Build output directory.
  if player_id is not None:
    out_dir = os.path.join(
        runner._output_dir, f'grpo_pass_{pass_idx}_p{player_id}'
    )
  else:
    out_dir = os.path.join(runner._output_dir, f'grpo_pass_{pass_idx}')

  runner._backend.model.train()

  # Ensure no leftover 'ref' adapter from a previous GRPOTrainer step.
  prev_adapter = (
      runner._backend.get_active_adapter()
      if hasattr(runner._backend, 'get_active_adapter')
      else getattr(runner._backend.model, 'active_adapter', 'default')
  )
  _cleanup_ref_adapter(runner._backend.model, prev_adapter)

  max_train_batch = 4
  candidates = [
      d
      for d in range(1, max_train_batch + 1)
      if runner._config.num_generations % d == 0
  ]
  batch_size = max(candidates) if candidates else 1
  gen_batch_size = runner._config.num_generations

  scale_mode = getattr(runner._config, 'grpo_scale_rewards', 'batch')
  scale_kwargs = _resolve_scale_rewards(trl_module, scale_mode)
  logging.info('GRPO advantage scaling: requested=%s resolved=%s',
               scale_mode, scale_kwargs.get('scale_rewards', '<default>'))

  training_args = trl_module.GRPOConfig(
      **scale_kwargs,
      output_dir=out_dir,
      num_train_epochs=runner._config.train_epochs,
      per_device_train_batch_size=batch_size,
      gradient_accumulation_steps=4,
      learning_rate=runner._config.lr,
      max_grad_norm=runner._config.max_grad_norm,
      logging_steps=1,
      save_strategy='no',
      max_completion_length=runner._config.max_completion_length,
      num_generations=runner._config.num_generations,
      generation_batch_size=gen_batch_size,
      beta=runner._config.kl_coeff,
      temperature=runner._current_temperature,
      report_to='none',
  )

  trainer = trl_module.GRPOTrainer(
      model=runner._backend.model,
      args=training_args,
      reward_funcs=reward_fn,
      processing_class=runner._backend.tokenizer,
      train_dataset=_build_prompt_dataset(unique_prompts),
  )
  # ── Constrained / strategic action generation ──
  # When enabled, wrap model.generate to force action diversity.
  original_generate = runner._backend.model.generate
  diversifier = None
  strategic_mode = getattr(
      runner._config, 'strategic_action_mode', 'substitute'
  )
  if runner._config.strategic_action_selection and strategic_mode == 'substitute':
    # Post-generation substitution.  Sample K freely, de-duplicate by
    # parsed action, then overwrite the freed slots with strategic
    # actions the group does not already contain.
    sub_ratio = getattr(
        runner._config, 'strategic_action_forced_ratio', 1.0
    )
    logging.info(
        'Enabling strategic action selection [mode=substitute]: sample K=%d, '
        'de-duplicate by parsed action, back-fill up to %d slots with '
        'strategic actions (safe plays, risky plays, smart discards, '
        'diverse hints, then any unexplored legal action)',
        runner._config.num_generations,
        int(runner._config.num_generations * sub_ratio),
    )
    diversifier = GroupDiversifier(
        tokenizer=runner._backend.tokenizer,
        runner=runner,
        num_generations=runner._config.num_generations,
        max_substitution_fraction=sub_ratio,
    )

    def wrapped_generate(*args, **kwargs):
      output = original_generate(*args, **kwargs)
      # TRL calls generate(prompt_ids, attention_mask=..., ...), so the
      # prompt width comes from the first positional argument.
      prompt_ids = args[0] if args else kwargs.get('input_ids')
      if prompt_ids is None or not torch.is_tensor(prompt_ids):
        logging.warning(
            '[diversify] could not locate prompt ids in generate() call; '
            'group left as sampled'
        )
        return output
      prompt_len = prompt_ids.shape[1]
      sequences = getattr(output, 'sequences', output)
      sequences = diversifier.diversify(sequences, prompt_len)
      if hasattr(output, 'sequences'):
        output.sequences = sequences
        return output
      return sequences

    runner._backend.model.generate = wrapped_generate
  elif runner._config.strategic_action_selection:
    logging.info(
        'Enabling strategic action selection [mode=logits]: state-aware '
        'action injection (safe plays, risky plays, smart discards, '
        'diverse hints) for K=%d',
        runner._config.num_generations,
    )

    def wrapped_generate(*args, **kwargs):
      from transformers.generation.logits_process import LogitsProcessorList  # pylint: disable=g-import-not-at-top

      processor = StrategicActionLogitsProcessor(
          tokenizer=runner._backend.tokenizer,
          runner=runner,
          num_generations=runner._config.num_generations,
          max_forced_fraction=getattr(
              runner._config, 'strategic_action_forced_ratio', 1.0
          ),
      )
      lp_list = kwargs.get('logits_processor', None)
      if lp_list is None:
        kwargs['logits_processor'] = LogitsProcessorList([processor])
      else:
        kwargs['logits_processor'] = LogitsProcessorList(
            list(lp_list) + [processor]
        )
      return original_generate(*args, **kwargs)

    runner._backend.model.generate = wrapped_generate
  elif runner._config.constrained_action_types:
    forced = max(1, runner._config.num_generations // 8)
    logging.info(
        'Enabling constrained action generation: %d forced per type '
        '(Play/Discard/Hint), %d free completions (K=%d)',
        forced,
        runner._config.num_generations - 3 * forced,
        runner._config.num_generations,
    )

    def wrapped_generate(*args, **kwargs):
      from transformers.generation.logits_process import LogitsProcessorList  # pylint: disable=g-import-not-at-top

      processor = ActionTypeLogitsProcessor(
          tokenizer=runner._backend.tokenizer,
          num_generations=runner._config.num_generations,
          forced_per_type=forced,
      )
      lp_list = kwargs.get('logits_processor', None)
      if lp_list is None:
        kwargs['logits_processor'] = LogitsProcessorList([processor])
      else:
        kwargs['logits_processor'] = LogitsProcessorList(
            list(lp_list) + [processor]
        )
      return original_generate(*args, **kwargs)

    runner._backend.model.generate = wrapped_generate

  try:
    trainer.train()
  finally:
    if (
        runner._config.strategic_action_selection
        or runner._config.constrained_action_types
    ):
      runner._backend.model.generate = original_generate
    if diversifier is not None:
      diversifier.log_summary(pass_idx, player_id)
    _cleanup_ref_adapter(runner._backend.model, prev_adapter)
    if hasattr(runner._backend, 'set_active_adapter') and prev_adapter:
      runner._backend.set_active_adapter(prev_adapter)

  # Extract training metrics.
  pass_loss = 0.0
  pass_reward = 0.0
  if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
    losses = [
        e['loss']
        for e in trainer.state.log_history
        if 'loss' in e and isinstance(e['loss'], (int, float))
    ]
    rew_vals = [
        e.get('reward', e.get('rewards/game_reward_fn/mean', 0.0))
        for e in trainer.state.log_history
        if 'reward' in e or 'rewards/game_reward_fn/mean' in e
    ]
    if losses:
      pass_loss = float(np.mean(losses))
    if rew_vals:
      pass_reward = float(np.mean(rew_vals))

  del trainer
  if torch.cuda.is_available():
    torch.cuda.empty_cache()

  player_label = f' (Player {player_id})' if player_id is not None else ''
  logging.info(
      'GRPO training step%s: loss=%.4f, reward=%.4f',
      player_label,
      pass_loss,
      pass_reward,
  )
  return pass_loss, pass_reward


# ═══════════════════════════════════════════════════════════════════════
# Main sampled GRPO loop
# ═══════════════════════════════════════════════════════════════════════


def run_sampled(runner) -> None:
  """Run the full sampled GRPO training loop.

  For each pass:
    1. Collect game-state prompts by playing episodes.
    2. Build a reward function using state simulation.
    3. Create a TRL GRPOTrainer and train for one epoch.
    4. Log training and eval metrics.
    5. Save a checkpoint.

  Args:
    runner: The ``GRPORunner`` instance.
  """
  from backend.gemma_backend import _lazy_import_hf  # pylint: disable=g-import-not-at-top

  _lazy_import_hf()
  import trl  # pylint: disable=g-import-not-at-top

  logging.info(
      'Starting GRPO training: %d passes, %d episodes/pass, K=%d, '
      'reward_sim=%s, horizon=%s, blend_w=%.2f, policy_turns=%d, '
      'turn_discount=%.3f, scale_rewards=%s, '
      'constrained_actions=%s, strategic_actions=%s, llm_partner=%s',
      runner._config.passes,
      runner._config.collect_episodes,
      runner._config.num_generations,
      runner._config.reward_simulation_mode,
      runner._config.truncated_rollout_horizon,
      runner._config.reward_blend_weight,
      runner._config.reward_policy_turns,
      runner._config.reward_turn_discount,
      runner._config.grpo_scale_rewards,
      runner._config.constrained_action_types,
      runner._config.strategic_action_selection,
      runner._config.llm_partner_response,
  )

  start_time = time.time()
  total_episodes_so_far = 0

  for pass_idx in range(1, runner._config.passes + 1):
    pass_start = time.time()
    logging.info('=== GRPO pass %d/%d ===', pass_idx, runner._config.passes)

    # Clear Hanabi state cache from previous pass (no-op for other games).
    try:
      from env.hanabi import hanabi_env as _hanabi_env  # pylint: disable=g-import-not-at-top

      _hanabi_env.clear_state_cache()
    except ImportError:
      pass

    # ── Temperature annealing ──
    if runner._config.temperature_anneal_end is not None:
      progress = (pass_idx - 1) / max(runner._config.passes - 1, 1)
      annealed_temp = runner._config.temperature + progress * (
          runner._config.temperature_anneal_end - runner._config.temperature
      )
      min_floor = getattr(runner._config, 'temperature_floor', 0.5)
      if min_floor is not None and annealed_temp < min_floor:
        if pass_idx == 1 or pass_idx % 5 == 0:
          logging.warning(
              'Temperature %.3f is below floor %.3f; clamping to floor to'
              ' preserve generation diversity',
              annealed_temp,
              min_floor,
          )
        annealed_temp = min_floor
      runner._current_temperature = annealed_temp
      logging.info(
          'Temperature annealed to %.3f (pass %d/%d)',
          runner._current_temperature,
          pass_idx,
          runner._config.passes,
      )

    # ── Epsilon annealing ──
    if runner._config.epsilon_anneal_end is not None:
      progress = (pass_idx - 1) / max(runner._config.passes - 1, 1)
      runner._current_epsilon = runner._config.epsilon + progress * (
          runner._config.epsilon_anneal_end - runner._config.epsilon
      )
    logging.info(
        'Epsilon-greedy: %.3f (pass %d/%d)',
        runner._current_epsilon,
        pass_idx,
        runner._config.passes,
    )

    # ── Snapshot LoRA weights for stable partner simulation ──
    runner._frozen_lora_state = {
        name: param.data.clone()
        for name, param in runner._backend.model.named_parameters()
        if param.requires_grad
    }

    # ── Step 1: Collect prompts ──
    runner._backend.model.eval()
    prompt_entries, collect_stats = collect_game_prompts(
        runner,
        num_episodes=runner._config.collect_episodes,
        pass_idx=pass_idx,
        start_time=start_time,
    )
    total_episodes_so_far += runner._config.collect_episodes

    # Record collection metrics to results/collection_metrics.csv
    results_dir = os.path.join(runner._output_dir, 'results')
    if os.path.exists(results_dir):
      collect_csv_path = os.path.join(results_dir, 'collection_metrics.csv')
      write_header = not os.path.exists(collect_csv_path)
      try:
        with open(collect_csv_path, 'a') as f:
          if write_header:
            f.write(
                'pass,episode,collect/mean_reward,collect/min_reward,'
                'collect/max_reward,collect/std_reward,num_episodes,'
                'num_prompts,elapsed_sec\n'
            )
          f.write(
              f"{pass_idx},{total_episodes_so_far},"
              f"{collect_stats['mean_reward']:.6f},"
              f"{collect_stats['min_reward']:.6f},"
              f"{collect_stats['max_reward']:.6f},"
              f"{collect_stats['std_reward']:.6f},"
              f"{collect_stats['num_episodes']},"
              f"{collect_stats['num_prompts']},"
              f"{time.time() - start_time:.1f}\n"
          )
      except IOError as e:
        logging.warning('Failed to write collection metrics CSV: %s', e)

    if not prompt_entries:
      logging.warning('No prompts collected in pass %d, skipping.', pass_idx)
      continue

    # ── Step 2–3: Train (per-player or combined) ──
    total_pass_loss = 0.0
    total_pass_reward = 0.0
    num_train_steps = 0

    if runner._config.per_player_updates:
      player_groups: dict[int, list[dict]] = {}
      for entry in prompt_entries:
        pid = entry['player_id']
        if pid not in player_groups:
          player_groups[pid] = []
        player_groups[pid].append(entry)

      for pid in sorted(player_groups.keys()):
        group_entries = player_groups[pid]
        unique_prompts = list({e['prompt'] for e in group_entries})
        if not unique_prompts:
          continue
        logging.info(
            'Pass %d: training on %d unique prompts for Player %d.',
            pass_idx,
            len(unique_prompts),
            pid,
        )
        p_loss, p_reward = _train_grpo_on_prompts(
            runner, unique_prompts, pass_idx, pid, trl
        )
        total_pass_loss += p_loss
        total_pass_reward += p_reward
        num_train_steps += 1
    else:
      unique_prompts = list({e['prompt'] for e in prompt_entries})
      logging.info(
          'Pass %d: %d unique prompts from %d total.',
          pass_idx,
          len(unique_prompts),
          len(prompt_entries),
      )
      p_loss, p_reward = _train_grpo_on_prompts(
          runner, unique_prompts, pass_idx, None, trl
      )
      total_pass_loss += p_loss
      total_pass_reward += p_reward
      num_train_steps += 1

    pass_elapsed = time.time() - pass_start
    avg_loss = total_pass_loss / num_train_steps if num_train_steps else 0.0
    avg_reward = total_pass_reward / num_train_steps if num_train_steps else 0.0
    logging.info(
        'GRPO pass %d complete in %.1f sec (loss=%.4f, reward=%.2f).',
        pass_idx,
        pass_elapsed,
        avg_loss,
        avg_reward,
    )

    # ── Step 4: Log training metrics ──
    if runner._log_training_step_fn is not None:
      runner._log_training_step_fn(pass_idx, avg_reward, avg_loss, start_time)

    # ── Step 5: Evaluate ──
    runner._backend.model.eval()
    eval_metrics = runner._evaluate_fn(runner._config.num_eval_episodes)
    eval_metrics['eval/collection_mean_reward'] = collect_stats['mean_reward']
    logging.info('--- Evaluation after GRPO pass %d ---', pass_idx)
    for k, v in sorted(eval_metrics.items()):
      logging.info('  %s: %.4f', k, v)

    if runner._log_eval_metrics_fn is not None:
      runner._log_eval_metrics_fn(total_episodes_so_far, eval_metrics)

    # ── Step 6: Checkpoint ──
    runner._save_checkpoint_fn(total_episodes_so_far)

  # ── Final summary and checkpoint ──
  total_time = time.time() - start_time
  logging.info(
      'GRPO training complete: %d passes (%d episodes) in %.1f sec.',
      runner._config.passes,
      total_episodes_so_far,
      total_time,
  )
  if runner._write_summary_fn is not None:
    runner._write_summary_fn(total_time)
  runner._save_checkpoint_fn(total_episodes_so_far, suffix='final')
