"""Sampled (TRL-based) GRPO runner.

This module implements the *sampled* GRPO variant that collects prompts
by playing episodes, then trains via TRL's ``GRPOTrainer``.  Suitable
for larger games where exhaustive game-tree enumeration is infeasible.

All public functions accept a ``runner`` parameter — the ``GRPORunner``
instance that holds shared state (env, backend, config, callbacks).
"""

from collections.abc import Mapping
import contextlib
import copy
import dataclasses
import json
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


from env import game_env
from env.hanabi.hanabi_env import deserialize_game_and_state
from env.hanabi.hanabi_env import serialize_game_and_state

# Aliases for internal and external compatibility.
_serialize_game_and_state = serialize_game_and_state
_deserialize_game_and_state = deserialize_game_and_state


# ═══════════════════════════════════════════════════════════════════════
# Prompt collection
# ═══════════════════════════════════════════════════════════════════════


def collect_game_prompts(
    runner,
    num_episodes: int,
    pass_idx: int = 1,
    start_time: float = 0.0,
) -> tuple[list[dict], dict]:
  """Collect game-state prompts by playing episodes with LLM agents.

  Batches episode rollouts in lockstep to utilize GPU tensor parallelism.
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
    A tuple of ``(all_prompts, collect_stats)`` where ``all_prompts`` is a
    list of prompt-entry dicts and ``collect_stats`` contains summary metrics.
  """
  all_prompts = []
  num_players = runner._game_config.num_players
  ep_rewards = []

  if num_episodes <= 0:
    return all_prompts, {
        'mean_reward': 0.0,
        'min_reward': 0.0,
        'max_reward': 0.0,
        'std_reward': 0.0,
        'num_episodes': 0,
        'num_prompts': 0,
    }

  is_graduated = getattr(runner, '_curriculum_graduated', False)
  window_size = getattr(runner._config, 'curriculum_window_size', 0)
  if is_graduated:
    max_horizon = 1000
    logging.info(
        '[curriculum collection] Pass %d: Curriculum graduated! Collecting'
        ' %d episodes to game completion (full fine-tuning mode)',
        pass_idx,
        num_episodes,
    )
  elif window_size > 0:
    max_horizon = getattr(
        runner._config, 'get_curriculum_horizon', lambda p: 1000
    )(pass_idx)
    start_turn, end_turn = runner._config.get_curriculum_active_window(pass_idx)
    logging.info(
        '[curriculum collection] Pass %d: collecting %d episodes up to'
        ' horizon turn %d (active window: turns [%d, %d))',
        pass_idx,
        num_episodes,
        max_horizon,
        start_turn,
        end_turn,
    )
  else:
    max_horizon = 1000
    logging.info(
        '[curriculum collection] Pass %d: collecting %d episodes to game'
        ' completion (no curriculum)',
        pass_idx,
        num_episodes,
    )

  target_batch_size = (
      getattr(runner._config, 'collect_batch_size', None) or num_episodes
  )
  logging.info(
      '[collection] Batched collection active: batch_size=%d for %d episodes',
      target_batch_size,
      num_episodes,
  )

  bot_partner_enabled = (
      getattr(runner._config, 'bot_partner', False) and num_players == 2
  )
  if bot_partner_enabled:
    if not hasattr(runner, '_heuristic_bot') or runner._heuristic_bot is None:
      bot_type = getattr(runner._config, 'bot_type', 'belief_lookahead')
      if bot_type == 'belief_lookahead':
        try:
          from env.hanabi.belief_expert import SafeBeliefLookaheadPlayer  # pylint: disable=g-import-not-at-top
          runner._heuristic_bot = SafeBeliefLookaheadPlayer(
              runner._env.game, n_worlds=1, seed=42
          )
        except Exception as e:
          logging.warning(
              'Failed to load SafeBeliefLookaheadPlayer: %s, falling back to SafePlayPlayer', e
          )
          try:
            from env.hanabi.heuristic_player import SafePlayPlayer  # pylint: disable=g-import-not-at-top
            runner._heuristic_bot = SafePlayPlayer(seed=42)
          except ImportError:
            runner._heuristic_bot = None
      else:
        try:
          from env.hanabi.heuristic_player import SafePlayPlayer  # pylint: disable=g-import-not-at-top
          runner._heuristic_bot = SafePlayPlayer(seed=42)
        except ImportError:
          runner._heuristic_bot = None

  episodes_collected = 0
  while episodes_collected < num_episodes:
    batch_episodes = min(target_batch_size, num_episodes - episodes_collected)
    global_ep_ids = [
        episodes_collected + b + 1 for b in range(batch_episodes)
    ]

    envs = [
        game_env.create_env(runner._game_config) for _ in range(batch_episodes)
    ]
    renderers = [
        [
            game_env.create_renderer(
                runner._game_config,
                max_history_turns=getattr(runner, 'max_history_turns', 20),
            )
            for _ in range(num_players)
        ]
        for _ in range(batch_episodes)
    ]
    time_steps = [env.reset() for env in envs]
    action_histories = [[] for _ in range(batch_episodes)]
    trajectories = [
        [PlayerTrajectory(player_id=p) for p in range(num_players)]
        for _ in range(batch_episodes)
    ]
    bot_players = [
        (1 if ep % 2 == 1 else 0) if bot_partner_enabled else None
        for ep in global_ep_ids
    ]

    active = list(range(batch_episodes))

    while active:
      # Step A: Advance heuristic bot turns until each active game needs the LLM or terminates.
      still_active = []
      for b_idx in active:
        env = envs[b_idx]
        ts = time_steps[b_idx]
        bot_p = bot_players[b_idx]

        while not ts.last():
          if len(action_histories[b_idx]) >= max_horizon:
            break
          cur_p = ts.current_player()
          if bot_p is not None and cur_p == bot_p:
            st = env._state
            legals_desc = renderers[b_idx][cur_p].render_legal_actions(
                st, cur_p, env.game
            )
            legals = [a for a, _ in legals_desc]
            if runner._heuristic_bot is not None:
              act_id = runner._heuristic_bot.select_action(st, cur_p, env.game)
            else:
              act_id = int(np.random.choice(legals))
            act_text = st.action_to_string(cur_p, act_id)
            st_text = renderers[b_idx][cur_p].render_state(st, cur_p, env.game)
            trajectories[b_idx][cur_p].steps.append(
                RLTrajectoryStep(
                    prompt='',
                    action_text=act_text,
                    action_id=act_id,
                    log_prob=0.0,
                    state_text=st_text,
                    llm_response=act_text,
                    game_action_text=act_text,
                )
            )
            action_histories[b_idx].append(act_id)
            ts = env.step([act_id])
            time_steps[b_idx] = ts
          else:
            break

        # Check termination or horizon cutoff for this game.
        if ts.last() or len(action_histories[b_idx]) >= max_horizon:
          if ts.rewards is not None:
            for p in range(num_players):
              trajectories[b_idx][p].reward = ts.rewards[p]
          elif hasattr(env._state, 'returns'):
            ret = env._state.returns()
            for p in range(num_players):
              trajectories[b_idx][p].reward = ret[p]

          m_r = float(np.mean([t.reward for t in trajectories[b_idx]]))
          ep_rewards.append(m_r)

          ep_num = global_ep_ids[b_idx]
          global_ep = (pass_idx - 1) * num_episodes + ep_num
          if runner._log_episode_fn is not None:
            runner._log_episode_fn(global_ep, trajectories[b_idx], 0.0, False)
          if runner._update_metrics_fn is not None:
            runner._update_metrics_fn(trajectories[b_idx], 0.0)

          ep_elapsed = time.time() - start_time if start_time > 0 else 0.0
          actions_summary = ' | '.join(
              f'P{t.player_id}:[{",".join(s.game_action_text for s in t.steps)}]'
              for t in trajectories[b_idx]
          )
          role_info = ''
          if bot_p is not None:
            role_info = f' (P{1-bot_p}:LLM vs P{bot_p}:Bot)'
          print(
              f'[pass {pass_idx} collect {ep_num}/{num_episodes}]{role_info} reward={m_r:.2f} '
              f'({ep_elapsed:.1f}s) {actions_summary}',
              flush=True,
          )
        else:
          still_active.append(b_idx)

      active = still_active
      if not active:
        break

      # Step B: Build prompts for all environments waiting on the LLM.
      batch_prompts = []
      batch_meta = []
      for b_idx in active:
        env = envs[b_idx]
        ts = time_steps[b_idx]
        cur_p = ts.current_player()
        st = env._state
        st_text = renderers[b_idx][cur_p].render_state(st, cur_p, env.game)
        legals_desc = renderers[b_idx][cur_p].render_legal_actions(
            st, cur_p, env.game
        )
        legals = [a for a, _ in legals_desc]
        descs = [d for _, d in legals_desc]
        prompt = runner._agents[cur_p]._build_prompt(st_text, legals, descs)
        batch_prompts.append(prompt)
        batch_meta.append((b_idx, cur_p, st_text, legals, legals_desc, prompt))

      # Step C: GPU batched generation.
      if hasattr(runner._backend, 'generate_batch'):
        responses = runner._backend.generate_batch(
            batch_prompts,
            temperature=runner._current_temperature,
            max_tokens=runner._config.max_completion_length,
        )
      else:
        responses = [
            runner._backend.generate(
                p,
                temperature=runner._current_temperature,
                max_tokens=runner._config.max_completion_length,
            )
            for p in batch_prompts
        ]

      # Step D: Parse and step each environment.
      for (b_idx, cur_p, st_text, legals, legals_desc, prompt), response in zip(
          batch_meta, responses
      ):
        env = envs[b_idx]
        st = env._state
        action_id = renderers[b_idx][cur_p].parse_action(response, legals_desc)
        if action_id is None:
          action_id = int(np.random.choice(legals))

        act_text = st.action_to_string(cur_p, action_id)

        prompt_entry = {
            'prompt': prompt,
            'player_id': cur_p,
            'action_history': list(action_histories[b_idx]),
            'legal_actions': legals,
            'legal_actions_desc': legals_desc,
            'state_text': st_text,
            'turn_index': len(action_histories[b_idx]),
            'episode': global_ep_ids[b_idx],
            'serialized_state': _serialize_game_and_state(env.game, st),
        }
        all_prompts.append(prompt_entry)
        runner._prompt_metadata[prompt] = prompt_entry

        trajectories[b_idx][cur_p].steps.append(
            RLTrajectoryStep(
                prompt=prompt,
                action_text=response.strip() if response else '',
                action_id=action_id,
                log_prob=0.0,
                state_text=st_text,
                llm_response=response or '',
                game_action_text=act_text,
            )
        )

        action_histories[b_idx].append(action_id)
        time_steps[b_idx] = env.step([action_id])

    episodes_collected += batch_episodes

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
  if hasattr(runner._backend, 'sync_replica'):
    runner._backend.sync_replica()

  runner._frozen_lora_depth = 1  # pylint: disable=protected-access
  try:
    yield
  finally:
    for name, param in runner._backend.model.named_parameters():  # pylint: disable=protected-access
      if name in live_lora_state:
        param.data.copy_(live_lora_state[name])
    if hasattr(runner._backend, 'sync_replica'):
      runner._backend.sync_replica()
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
      legal = state.legal_actions(state.current_player())
      if action not in legal:
        action = int(np.random.choice(legal)) if legal else None
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
  bot_type = getattr(runner._config, 'bot_type', 'belief_lookahead')
  heuristic = None
  game = getattr(runner._env, 'game', None)
  if bot_type == 'belief_lookahead':
    try:
      from env.hanabi.belief_expert import SafeBeliefLookaheadPlayer  # pylint: disable=g-import-not-at-top
      heuristic = SafeBeliefLookaheadPlayer(game, n_worlds=1, seed=seed or 42)
    except Exception:
      heuristic = None
  if heuristic is None:
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
      if action is None or action not in legal:
        action = int(rng.choice(legal)) if legal else None
    else:
      action = int(rng.choice(legal)) if legal else None
    if action is None:
      break
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

      _, state = deserialize_game_and_state(ser_state)

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
          completion_text = f'Reasoning: Selecting {desc}.\nAction: {desc}'
      elif is_reasoning:
        completion_text = f'Reasoning: Selecting {desc}.\nAction: {desc}'
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


def _strip_bos(prompt: str, tokenizer) -> str:
  """Removes the BOS text that ``_build_prompt_dataset`` may prepend."""
  bos = getattr(tokenizer, 'bos_token', None)
  if bos and isinstance(prompt, str) and prompt.startswith(bos):
    return prompt[len(bos) :]
  return prompt


def _build_prompt_dataset(prompts: list[str], tokenizer=None):
  """Build a HuggingFace Dataset from prompt strings.

  TRL tokenizes plain-string prompts with ``add_special_tokens=False``, which
  drops the BOS token that BC training and the backend's own generation (eval,
  rollouts, partner moves) put in front of every prompt.  The BOS is therefore
  written into the prompt text, provided the tokenizer maps it back to exactly
  the token IDs the backend produces.  ``reward_fn`` undoes this with
  ``_strip_bos``.

  Args:
    prompts: Raw prompt strings.
    tokenizer: The policy tokenizer, or None to leave the prompts unchanged.

  Returns:
    A ``datasets.Dataset`` with a single ``prompt`` column.
  """
  from datasets import Dataset  # pylint: disable=g-import-not-at-top

  bos = getattr(tokenizer, 'bos_token', None)
  if bos and prompts:
    with_bos = tokenizer(bos + prompts[0], add_special_tokens=False)
    if with_bos['input_ids'] == tokenizer(prompts[0])['input_ids']:
      prompts = [p if p.startswith(bos) else bos + p for p in prompts]
      logging.info('[GRPO] Prepended BOS %r to TRL prompts.', bos)
    else:
      logging.warning(
          '[GRPO] BOS-prefixed prompt text does not reproduce the backend'
          ' tokenization; TRL prompts left unchanged (no BOS).'
      )
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

def _compute_action_reward(
    runner,
    prompt_text: str,
    action_id: int,
    parsed: bool,
    p_id: int,
    ser_state: str | None,
    action_history: list[int],
    pass_idx: int,
    partner_action: int | None = None,
    post_action_state=None,
    post_policy_state=None,
    policy_actions: list[int] | None = None,
) -> float:
  """Compute the scalar reward for an action given the runner's simulation config."""
  if not parsed:
    return -0.3

  sim_mode = runner._config.reward_simulation_mode

  if sim_mode == 'dense':
    from learn.action_reward import evaluate_action_quality  # pylint: disable=g-import-not-at-top
    if ser_state is not None:
      _, eval_state = deserialize_game_and_state(ser_state)
    else:
      runner._env.reset()
      eval_state = runner._env._state
      for a in action_history:
        if eval_state.is_terminal():
          break
        eval_state.apply_action(a)
    return float(evaluate_action_quality(eval_state, action_id, p_id))

  if sim_mode == 'dense_chain':
    from learn.action_reward import evaluate_dense_chain  # pylint: disable=g-import-not-at-top
    blend_w = runner._config.reward_blend_weight

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
          policy_actions=policy_actions,
          post_action_state=post_action_state.clone()
          if post_action_state is not None and hasattr(post_action_state, 'clone')
          else None,
          post_policy_state=post_policy_state.clone()
          if post_policy_state is not None and hasattr(post_policy_state, 'clone')
          else None,
      )

    if blend_w > 0:
      if post_policy_state is not None and hasattr(post_policy_state, 'clone'):
        blend_state = post_policy_state.clone()
        runner._env.set_state(blend_state)
      else:
        if post_action_state is not None and hasattr(post_action_state, 'clone'):
          blend_state = post_action_state.clone()
          runner._env.set_state(blend_state)
        elif ser_state is not None:
          _, blend_state = deserialize_game_and_state(ser_state)
          runner._env.set_state(blend_state)
          if not blend_state.is_terminal():
            legal0 = blend_state.legal_actions(blend_state.current_player())
            if action_id in legal0:
              blend_state.apply_action(action_id)
            elif legal0:
              blend_state.apply_action(int(np.random.choice(legal0)))
        else:
          runner._env.reset()
          blend_state = runner._env._state
          for a in action_history:
            if blend_state.is_terminal():
              break
            blend_state.apply_action(a)
          blend_state = runner._env._state
          if not blend_state.is_terminal():
            legal0 = blend_state.legal_actions(blend_state.current_player())
            if action_id in legal0:
              blend_state.apply_action(action_id)
            elif legal0:
              blend_state.apply_action(int(np.random.choice(legal0)))

        policy_turns = max(
            runner._config.reward_policy_turns,
            2 if getattr(runner._config, 'llm_partner_response', False) else 1,
        )
        if not blend_state.is_terminal():
          _rollout_policy_turns(runner, blend_state, policy_turns - 1)

      survival_exp = runner._config.reward_survival_exponent
      lives_after = None
      if survival_exp > 0 and hasattr(blend_state, 'life_tokens'):
        lives_after = blend_state.life_tokens()

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
          game_score_norm -= (1.0 - life_fraction) * 0.3

      return float((1 - blend_w) * primary_reward + blend_w * game_score_norm)

    return float(primary_reward)

  if (
      runner._config.reward_num_simulations > 1
      and runner._config.reward_variance_penalty > 0
  ):
    sim_rewards = [
        simulate_from_state(runner, action_history, action_id, p_id, ser_state)
        for _ in range(runner._config.reward_num_simulations)
    ]
    mean_r = float(np.mean(sim_rewards))
    std_r = float(np.std(sim_rewards))
    return float(mean_r - runner._config.reward_variance_penalty * std_r)

  return float(simulate_from_state(runner, action_history, action_id, p_id, ser_state))


def _log_group_records(group_records: list[dict], group_counter: list[int]) -> None:
  """Log per-group reward distributions and advantage statistics."""
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
        '  *** DEGENERATE: std=0, zero gradient ***' if std_r < 1e-6 else '',
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

  results_dir = os.path.join(runner._output_dir, 'results')
  os.makedirs(results_dir, exist_ok=True)
  reward_cache_path = os.path.join(
      results_dir, f'reward_cache_pass{pass_idx}.json'
  )
  persisted_rewards: dict[tuple[str, int], float] = {}
  persisted_group_records: list[dict] = []
  if os.path.exists(reward_cache_path):
    try:
      with open(reward_cache_path, 'r') as f:
        rc_data = json.load(f)
      for k_str, val in rc_data.get('rewards', {}).items():
        if '|||' in k_str:
          p_txt, a_str = k_str.rsplit('|||', 1)
          persisted_rewards[(p_txt, int(a_str))] = float(val)
      if persisted_rewards:
        logging.info(
            '[REWARD CACHE] Loaded %d cached decision point evals for pass %d'
            ' from %s',
            len(persisted_rewards),
            pass_idx,
            reward_cache_path,
        )
    except Exception as e:
      logging.warning(
          '[REWARD CACHE] Failed to load %s: %s', reward_cache_path, e
      )

  interim_progress_path = os.path.join(results_dir, 'interim_progress.json')

  def _write_interim_progress(status_str: str = 'in_progress') -> dict:
    now = time.time()
    prev_ep = (pass_idx - 1) * runner._config.collect_episodes
    progress_meta = {
        'status': status_str,
        'pass_idx': pass_idx,
        'player_id': player_id,
        'completed_players': list(
            getattr(runner, '_pass_completed_players', [])
        ),
        'completed_prompts': int(
            getattr(runner, '_pass_completed_prompts', 0)
        ),
        'total_episodes': prev_ep,
        'cached_decision_evals': len(persisted_rewards),
        'timestamp': now,
    }
    try:
      with open(interim_progress_path, 'w') as f:
        json.dump(progress_meta, f, indent=2)
    except Exception as e:
      logging.warning(
          '[INTERIM PROGRESS] Failed to write %s: %s', interim_progress_path, e
      )
    return progress_meta

  def _flush_reward_cache_and_maybe_checkpoint(
      reward_cache_dict: dict,
      curr_group_records: list[dict],
      force_save_file: bool = False,
  ) -> None:
    del curr_group_records
    now = time.time()
    interval_sec = (
        getattr(runner._config, 'checkpoint_interval_minutes', 10.0) * 60.0
    )
    if interval_sec <= 0:
      interval_sec = 600.0
    last_flush = getattr(runner, '_last_cache_flush_time', 0.0)
    if force_save_file or (now - last_flush >= 60.0):
      try:
        for (p_txt, act_id), r_tensor in reward_cache_dict.items():
          persisted_rewards[(p_txt, int(act_id))] = float(
              r_tensor.item() if hasattr(r_tensor, 'item') else r_tensor
          )
        serializable_rewards = {
            f'{p_txt}|||{act_id}': val
            for (p_txt, act_id), val in persisted_rewards.items()
        }
        with open(reward_cache_path, 'w') as f:
          json.dump(
              {
                  'pass_idx': pass_idx,
                  'player_id': player_id,
                  'rewards': serializable_rewards,
                  'timestamp': now,
              },
              f,
          )
        runner._last_cache_flush_time = now
      except Exception as e:
        logging.warning(
            '[REWARD CACHE] Failed to write %s: %s', reward_cache_path, e
        )

    last_ckpt = getattr(runner, '_last_periodic_save_time', 0.0)
    if last_ckpt <= 0.0:
      runner._last_periodic_save_time = now
    elif now - last_ckpt >= interval_sec:
      runner._last_periodic_save_time = now
      prev_ep = (pass_idx - 1) * runner._config.collect_episodes
      interim_meta = _write_interim_progress('in_progress')
      logging.info(
          '[PERIODIC CHECKPOINT] Saving interim checkpoint and syncing %d'
          ' decision point evals (pass %d, player %s)',
          len(persisted_rewards),
          pass_idx,
          player_id,
      )
      runner._save_checkpoint_fn(prev_ep, suffix='interim', metadata=interim_meta)

  def reward_fn(completions, prompts=None, **kwargs):
    del kwargs
    if prompts is not None:
      # Metadata and reward caches are keyed by the raw prompt, without the
      # BOS that _build_prompt_dataset prepends for TRL.
      prompts = [_strip_bos(p, runner._backend.tokenizer) for p in prompts]
    rewards = []
    reward_cache = {
        k: torch.tensor(float(v)) for k, v in persisted_rewards.items()
    }
    group_records = []  # one dict per completion, for the group/z-score dump
    _action_type_tracker = {}  # prompt_text -> {play, discard, hint, total}
    cache_hits = 0
    partner_actions_by_key = {}
    post_action_states_by_key = {}
    post_policy_states_by_key = {}
    policy_actions_by_key = {}
    effective_policy_turns = max(
        runner._config.reward_policy_turns,
        2 if getattr(runner._config, 'llm_partner_response', False) else 1,
    )
    if effective_policy_turns >= 2:
      needed_queries = {}
      for i, completion in enumerate(completions):
        prompt_text = (
            prompts[i] if prompts is not None and i < len(prompts) else ''
        )
        metadata = runner._prompt_metadata.get(prompt_text, {})
        legal_actions_desc = metadata.get('legal_actions_desc', [])
        p_id = metadata.get('player_id', 0)
        ser_state = metadata.get('serialized_state', None)
        action_history = metadata.get('action_history', [])

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
        if action_id is not None:
          cache_key = (prompt_text, action_id)
          if cache_key not in needed_queries and cache_key not in reward_cache:
            needed_queries[cache_key] = (
                ser_state, action_history, action_id, p_id
            )

      if needed_queries:
        policy_turns = effective_policy_turns

        # Step 1: Apply Turn 1 (the candidate action_id) to initialize states.
        curr_states = {}
        for key, (ser_state, action_history, action_id, p_id) in needed_queries.items():
          policy_actions_by_key[key] = []
          if ser_state is not None:
            _, step_state = _deserialize_game_and_state(ser_state)
          else:
            runner._env.reset()
            step_state = runner._env._state
            for a in action_history:
              if step_state.is_terminal():
                break
              step_state.apply_action(a)
            step_state = runner._env._state

          if not step_state.is_terminal():
            legal0 = step_state.legal_actions(step_state.current_player())
            if action_id in legal0:
              step_state.apply_action(action_id)
            elif legal0:
              step_state.apply_action(int(np.random.choice(legal0)))

          if hasattr(step_state, 'clone'):
            post_action_states_by_key[key] = step_state.clone()
            curr_states[key] = step_state.clone()
          else:
            curr_states[key] = step_state

        # Step 2: Iteratively batch policy turns 2 .. policy_turns
        for turn_step in range(policy_turns - 1):
          batch_keys = []
          batch_prompts = []
          batch_legal_descs = []
          batch_pids = []

          for key, st in curr_states.items():
            if st.is_terminal():
              continue
            pid = st.current_player()
            legal = st.legal_actions(pid)
            if not legal:
              continue

            state_text = runner._renderers[pid].render_state(
                st, pid, runner._env.game
            )
            legal_actions_with_desc = runner._renderers[pid].render_legal_actions(
                st, pid, runner._env.game
            )
            legal_actions = [a for a, _ in legal_actions_with_desc]
            action_descriptions = [d for _, d in legal_actions_with_desc]

            partner_prompt = runner._agents[pid]._build_prompt(
                state_text, legal_actions, action_descriptions
            )
            batch_keys.append(key)
            batch_prompts.append(partner_prompt)
            batch_legal_descs.append(legal_actions_with_desc)
            batch_pids.append(pid)

          if not batch_prompts:
            break

          with _frozen_lora_active(runner):
            if hasattr(runner._backend, 'generate_batch'):
              responses = runner._backend.generate_batch(
                  batch_prompts,
                  temperature=runner._current_temperature,
                  max_tokens=runner._config.max_completion_length,
              )
            else:
              responses = [
                  runner._backend.generate(
                      p,
                      temperature=runner._current_temperature,
                      max_tokens=runner._config.max_completion_length,
                  )
                  for p in batch_prompts
              ]

          for k_idx, key in enumerate(batch_keys):
            resp = responses[k_idx]
            descs = batch_legal_descs[k_idx]
            pid = batch_pids[k_idx]
            p_act = runner._renderers[pid].parse_action(resp, descs)
            legals = [a for a, _ in descs]
            if p_act is None or p_act not in legals:
              p_act = int(np.random.choice(legals)) if legals else None

            policy_actions_by_key[key].append(p_act)
            if turn_step == 0:
              partner_actions_by_key[key] = p_act

            if p_act is not None and not curr_states[key].is_terminal():
              curr_states[key].apply_action(p_act)

        for key, st in curr_states.items():
          if hasattr(st, 'clone'):
            post_policy_states_by_key[key] = st.clone()

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

      reward = _compute_action_reward(
          runner=runner,
          prompt_text=prompt_text,
          action_id=action_id,
          parsed=parsed,
          p_id=p_id,
          ser_state=ser_state,
          action_history=action_history,
          pass_idx=pass_idx,
          partner_action=partner_actions_by_key.get(cache_key, None),
          post_action_state=post_action_states_by_key.get(cache_key, None),
          post_policy_state=post_policy_states_by_key.get(cache_key, None),
          policy_actions=policy_actions_by_key.get(cache_key, None),
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
      _flush_reward_cache_and_maybe_checkpoint(reward_cache, group_records)

    if cache_hits > 0:
      logging.info(
          'Reward cache: %d/%d hits (%.0f%% duplicates avoided)',
          cache_hits,
          len(rewards),
          100.0 * cache_hits / len(rewards),
      )

    _flush_reward_cache_and_maybe_checkpoint(
        reward_cache, group_records, force_save_file=True
    )
    persisted_group_records.extend(group_records)
    _log_group_records(group_records, group_counter)
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

  max_train_batch = getattr(runner._config, 'train_batch_size', 4) or 4
  candidates = [
      d
      for d in range(1, max_train_batch + 1)
      if runner._config.num_generations % d == 0
  ]
  batch_size = max(candidates) if candidates else 1

  # Determine prompt batch size: how many unique prompts are batched together
  # for parallel LLM generation.
  k = runner._config.num_generations
  cfg_p_batch = getattr(runner._config, 'prompt_batch_size', None)
  if cfg_p_batch is not None and cfg_p_batch > 0:
    p_batch = min(cfg_p_batch, len(unique_prompts))
  else:
    p_batch = min(len(unique_prompts), 40)
  p_batch = max(1, p_batch)

  # Pad unique_prompts so RepeatSampler (which drops len(chunk) != p_batch)
  # does not drop any leftover prompts.
  training_prompts = list(unique_prompts)
  remainder = len(training_prompts) % p_batch
  if remainder != 0:
    needed = p_batch - remainder
    training_prompts.extend(
        [unique_prompts[i % len(unique_prompts)] for i in range(needed)]
    )
    logging.info(
        'Padded prompts from %d to %d to be an exact multiple of'
        ' prompt_batch_size=%d',
        len(unique_prompts),
        len(training_prompts),
        p_batch,
    )

  gen_batch_size = p_batch * k
  grad_accum = max(1, gen_batch_size // batch_size)
  logging.info(
      'GRPO batch configuration: %d unique prompts -> %d training prompts, '
      'prompt_batch_size=%d, num_generations=%d, generation_batch_size=%d, '
      'train_batch_size=%d, grad_accum=%d',
      len(unique_prompts),
      len(training_prompts),
      p_batch,
      k,
      gen_batch_size,
      batch_size,
      grad_accum,
  )

  scale_mode = getattr(runner._config, 'grpo_scale_rewards', 'batch')
  scale_kwargs = _resolve_scale_rewards(trl_module, scale_mode)
  max_prompt_len = getattr(runner._config, 'max_seq_len', 2048) or 2048
  try:
    trl_fields = {f.name for f in dataclasses.fields(trl_module.GRPOConfig)}
    if 'max_prompt_length' in trl_fields:
      scale_kwargs['max_prompt_length'] = max_prompt_len
    if 'delta' in trl_fields:
      scale_kwargs['delta'] = 2.0
  except (TypeError, AttributeError):
    pass

  training_args = trl_module.GRPOConfig(
      **scale_kwargs,
      output_dir=out_dir,
      num_train_epochs=runner._config.train_epochs,
      per_device_train_batch_size=batch_size,
      gradient_accumulation_steps=grad_accum,
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

  class _PatchedGRPOTrainer(trl_module.GRPOTrainer):
    """GRPOTrainer that chunks inputs in _get_per_token_logps and clamps logp ratios."""

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
      self._active_inputs = inputs
      try:
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
      finally:
        self._active_inputs = None

    def _get_per_token_logps(
        self, model, input_ids, attention_mask, logits_to_keep, batch_size=None
    ):
      batch_size = batch_size or getattr(self.args, 'per_device_train_batch_size', 8) or 8
      logps = super()._get_per_token_logps(
          model, input_ids, attention_mask, logits_to_keep, batch_size=batch_size
      )
      active_inputs = getattr(self, '_active_inputs', None)
      if active_inputs is not None:
        for key, bound in (('ref_per_token_logps', 4.0), ('old_per_token_logps', 1.0)):
          target = active_inputs.get(key)
          if target is not None and target.shape == logps.shape:
            active_inputs[key] = torch.clamp(
                target, min=logps.detach() - bound, max=logps.detach() + bound
            )
      return logps

  trainer = _PatchedGRPOTrainer(
      model=runner._backend.model,
      args=training_args,
      reward_funcs=reward_fn,
      processing_class=runner._backend.tokenizer,
      train_dataset=_build_prompt_dataset(
          training_prompts, runner._backend.tokenizer
      ),
  )

  try:
    import transformers  # pylint: disable=g-import-not-at-top

    base_completed_prompts = int(
        getattr(runner, '_pass_completed_prompts', 0)
    )

    class _PeriodicCheckpointCallback(transformers.TrainerCallback):
      """Saves periodic interim checkpoints on optimizer step boundaries."""

      def on_step_end(self, args, state, control, **kwargs):
        del args, control, kwargs
        completions_done = state.global_step * batch_size * grad_accum
        gen_batches_done = completions_done // gen_batch_size
        runner._pass_completed_prompts = min(
            base_completed_prompts + len(unique_prompts),
            base_completed_prompts + gen_batches_done * p_batch,
        )
        _flush_reward_cache_and_maybe_checkpoint({}, [])

    trainer.add_callback(_PeriodicCheckpointCallback())
  except Exception as e:
    logging.warning('Failed to attach _PeriodicCheckpointCallback: %s', e)
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

  trainable = {
      n: p for n, p in runner._backend.model.named_parameters() if p.requires_grad
  }
  w_before = {n: p.detach().clone() for n, p in trainable.items()}
  if not hasattr(runner, '_lora_init'):
    runner._lora_init = w_before
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
    if hasattr(runner._backend, 'sync_replica'):
      runner._backend.sync_replica()

  # Extract training metrics and per-pass diagnostics (loss decomposition,
  # clip fraction, grad norm, LoRA weight movement).
  history = getattr(getattr(trainer, 'state', None), 'log_history', []) or []

  def _mean_of(key):
    vals = [e[key] for e in history if isinstance(e.get(key), (int, float))]
    return float(np.mean(vals)) if vals else float('nan')

  pass_loss = _mean_of('loss')
  pass_reward = _mean_of('reward')
  if np.isnan(pass_reward):
    pass_reward = _mean_of('rewards/game_reward_fn/mean')
  if np.isnan(pass_loss):
    pass_loss = 0.0
  if np.isnan(pass_reward):
    pass_reward = 0.0

  def _l2(deltas):
    if not deltas:
      return 0.0
    return float(torch.sqrt(sum((d.float() ** 2).sum() for d in deltas)))

  kl_term = runner._config.kl_coeff * _mean_of('kl')
  with torch.no_grad():
    diag = {
        'train/kl': _mean_of('kl'),
        'train/kl_term': kl_term,
        'train/policy_loss': pass_loss - kl_term,
        'train/clip_frac': _mean_of('clip_ratio/region_mean'),
        'train/grad_norm': _mean_of('grad_norm'),
        'train/completion_len': _mean_of('completions/mean_length'),
        'train/frac_zero_std_groups': _mean_of('frac_reward_zero_std'),
        'train/lora_step_delta': _l2(
            [p.detach() - w_before[n] for n, p in trainable.items()]
        ),
        'train/lora_total_drift': _l2([
            p.detach() - runner._lora_init[n]
            for n, p in trainable.items()
            if n in runner._lora_init
        ]),
        'train/lora_norm': _l2([p.detach() for p in trainable.values()]),
    }
  runner._train_diags = getattr(runner, '_train_diags', []) + [diag]
  logging.info(
      'GRPO pass diagnostics: %s', {k: round(v, 5) for k, v in diag.items()}
  )

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


def _eval_seed_base(runner) -> int | None:
  """Returns the fixed-deal base seed, or ``None`` if seeding is disabled."""
  seed = getattr(runner._config, 'eval_seed', -1)
  if seed is None or int(seed) < 0:
    return None
  return int(seed)


def _run_eval(runner, num_episodes: int, **kwargs) -> dict[str, float]:
  """Calls ``runner._evaluate_fn`` tolerating older ``evaluate`` signatures.

  Args:
    runner: The ``GRPORunner`` instance.
    num_episodes: Number of evaluation episodes.
    **kwargs: Forwarded to ``evaluate`` (``eval_llm_max_horizon``,
      ``seed_base``, ``metric_prefix``).

  Returns:
    Metrics dict.  If the bound evaluate() predates ``metric_prefix`` the
    legacy result keys are re-prefixed so callers can still merge rows.
  """
  try:
    return runner._evaluate_fn(num_episodes, **kwargs)
  except TypeError as e:
    logging.warning(
        '[evaluate] evaluate() rejected kwargs %s (%s); retrying with the'
        ' legacy signature (unseeded deals).',
        sorted(kwargs),
        e,
    )
    legacy = {
        k: v for k, v in kwargs.items() if k == 'eval_llm_max_horizon'
    }
    metrics = runner._evaluate_fn(num_episodes, **legacy)
    prefix = kwargs.get('metric_prefix', 'eval')
    if prefix != 'eval':
      metrics = {
          (prefix + k[len('eval'):] if k.startswith('eval/') else k): v
          for k, v in metrics.items()
      }
    return metrics


def _run_full_selfplay_eval(runner, pass_idx: int) -> dict[str, float]:
  """Full self-play evaluation on the fixed deals (``eval_full/*`` keys).

  The LLM plays *every* turn (no bot hand-off), so this measures whole-game
  strength independently of the curriculum horizon.  Returns ``{}`` when
  ``eval_full_episodes <= 0``.

  Args:
    runner: The ``GRPORunner`` instance.
    pass_idx: Current pass index (for logging only).

  Returns:
    Metrics dict with ``eval_full/*`` keys plus ``eval_full/elapsed_sec``.
  """
  n = int(getattr(runner._config, 'eval_full_episodes', 0) or 0)
  if n <= 0:
    return {}
  seed_base = _eval_seed_base(runner)
  logging.info(
      '[full self-play evaluation] Pass %d: %d episodes, LLM plays every turn'
      ' (no bot hand-off), deals=%s.',
      pass_idx,
      n,
      f'fixed (seeds {seed_base}..{seed_base + n - 1})'
      if seed_base is not None
      else 'random',
  )
  t0 = time.time()
  metrics = _run_eval(
      runner,
      n,
      eval_llm_max_horizon=None,
      seed_base=seed_base,
      metric_prefix='eval_full',
  )
  metrics['eval_full/elapsed_sec'] = time.time() - t0
  return metrics


_EVAL_MODES = ('full', 'bot_guided', 'both')


def _eval_flavours(eval_mode: str | None) -> tuple[bool, bool]:
  """Maps ``GRPOConfig.eval_mode`` to ``(run_bot_guided, run_full)``.

  Args:
    eval_mode: ``'full'`` (fixed-deal full self-play only -- the default),
      ``'bot_guided'`` (curriculum eval only: LLM plays to the horizon, the
      heuristic bot finishes the game) or ``'both'``.  ``None`` means
      ``'full'``.

  Returns:
    ``(run_bot_guided, run_full)`` booleans.

  Raises:
    ValueError: If ``eval_mode`` is not one of ``_EVAL_MODES``.
  """
  mode = str(eval_mode or 'full').strip().lower()
  if mode not in _EVAL_MODES:
    raise ValueError(
        f'Unsupported eval_mode={eval_mode!r}; expected one of {_EVAL_MODES}.'
    )
  return mode in ('bot_guided', 'both'), mode in ('full', 'both')


# ═══════════════════════════════════════════════════════════════════════
# GRPO sample-budget counters (x-axes of the decision-point dashboards)
# ═══════════════════════════════════════════════════════════════════════

# Each counter is logged with the pass's eval row as ``grpo/<name>`` (this
# pass) and ``grpo/<name>_total`` (cumulative since pass 1), so it reaches
# eval_metrics.csv, the checkpoint metadata and S2.  The launcher's second
# Flatboard dashboard plots ``eval_full.*`` against ``grpo.<name>_total``
# instead of the pass index: the curriculum settings decide how many decision
# points a pass trains on, so per-pass curves are not compute-matched across
# sweep arms.
#
#   decision_points: unique decision points GRPO expanded into a group this
#     pass (``len(unique_prompts)``, summed over per-player updates).
#   rollouts: nominal reward-simulated candidate rollouts, i.e.
#     ``decision_points * num_generations * train_epochs``.  The reward cache,
#     strategic de-duplication and prompt-batch padding make the exact number
#     of simulations differ slightly.
#   collected_decision_points: decision points visited while collecting the
#     pass's episodes (``collect_stats['num_prompts']``), trained on or not.
_GRPO_BUDGET_KEYS = ('decision_points', 'rollouts', 'collected_decision_points')


def _grpo_budget_metrics(
    pass_counts: Mapping[str, int], totals: Mapping[str, int]
) -> dict[str, int]:
  """Builds the ``grpo/*`` and ``grpo/*_total`` metrics for one eval row.

  Args:
    pass_counts: This pass's counts keyed by ``_GRPO_BUDGET_KEYS`` (missing
      keys count as 0).
    totals: Cumulative counts including this pass (missing keys count as 0).

  Returns:
    ``{'grpo/<key>': pass_count, 'grpo/<key>_total': total, ...}``.
  """
  metrics: dict[str, int] = {}
  for key in _GRPO_BUDGET_KEYS:
    metrics[f'grpo/{key}'] = int(pass_counts.get(key, 0))
    metrics[f'grpo/{key}_total'] = int(totals.get(key, 0))
  return metrics


def _load_grpo_budget_totals(
    results_dir: str, upto_episode: int
) -> dict[str, int]:
  """Restores the cumulative ``grpo/*_total`` counters of a resumed run.

  Unlike ``total_episodes_so_far`` the totals cannot be re-derived from the
  pass index, so they are read back from the last completed pass's row of
  ``results/eval_metrics.csv`` (synced down from CNS by ``RLTrainer``).  A
  run that predates these counters restarts them at zero with a warning.

  Args:
    results_dir: Local results directory holding ``eval_metrics.csv``.
    upto_episode: Episode count of the last completed pass; later rows (from
      an explicit ``--grpo_initial_pass`` rewind) are ignored.

  Returns:
    ``{key: total}`` for every key in ``_GRPO_BUDGET_KEYS``.
  """
  totals = {key: 0 for key in _GRPO_BUDGET_KEYS}
  csv_path = os.path.join(results_dir, 'eval_metrics.csv')
  try:
    # The trainer owns the CSV format; it is already imported by the time
    # GRPO runs, so this is a sys.modules lookup rather than a new import.
    from trainer.rl_trainer import read_eval_csv_row  # pylint: disable=g-import-not-at-top

    row = read_eval_csv_row(
        csv_path,
        [f'grpo/{key}_total' for key in _GRPO_BUDGET_KEYS],
        upto_episode,
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.warning(
        '[AUTO-RESUME] Could not read GRPO budget totals from %s: %s',
        csv_path,
        e,
    )
    row = {}
  if not row:
    logging.warning(
        '[AUTO-RESUME] No grpo/*_total columns in %s up to episode %d; the'
        ' cumulative decision-point counters restart at 0, so this work'
        " unit's curves are shifted left on the decision-point dashboard.",
        csv_path,
        upto_episode,
    )
    return totals
  for key in _GRPO_BUDGET_KEYS:
    totals[key] = int(round(row[f'grpo/{key}_total']))
  logging.info(
      '[AUTO-RESUME] Restored GRPO budget totals up to episode %d: %s',
      upto_episode,
      totals,
  )
  return totals


def _maybe_run_baselines(runner, results_dir: str) -> None:
  """Runs the one-off reference evaluations on the fixed deals (pass 0).

  Two anchors are logged at ``episode=0`` in ``eval_metrics.csv``:

  * ``eval_full/*`` -- the *initial* adapter (e.g. the BC checkpoint) in full
    self-play.  Without it, later ``eval_full`` rows cannot claim improvement.
  * ``eval_bot/*``  -- the heuristic hand-off bot in both seats for every turn
    (``eval_llm_max_horizon=0``).  Sizes the gap to the bot on the same deals.

  Results are cached in ``results/eval_baselines.json`` so a resumed run
  (``initial_pass > 1`` or an existing cache file) never repeats them.

  Args:
    runner: The ``GRPORunner`` instance.
    results_dir: Local results directory.
  """
  n = int(getattr(runner._config, 'eval_full_episodes', 0) or 0)
  if n <= 0:
    return
  cache_path = os.path.join(results_dir, 'eval_baselines.json')
  if os.path.exists(cache_path):
    logging.info(
        '[baselines] %s already exists; skipping baseline evaluation.',
        cache_path,
    )
    return

  seed_base = _eval_seed_base(runner)
  deals = (
      f'fixed (seeds {seed_base}..{seed_base + n - 1})'
      if seed_base is not None
      else 'random'
  )
  runner._backend.model.eval()
  merged: dict[str, float] = {}
  t0 = time.time()

  logging.info(
      '[baselines] Pass 0: initial adapter, full self-play, %d episodes,'
      ' deals=%s.',
      n,
      deals,
  )
  merged.update(
      _run_eval(
          runner,
          n,
          eval_llm_max_horizon=None,
          seed_base=seed_base,
          metric_prefix='eval_full',
      )
  )
  if getattr(runner._config, 'eval_bot_baseline', True):
    logging.info(
        '[baselines] Pass 0: heuristic bot in both seats for every turn,'
        ' %d episodes, deals=%s.',
        n,
        deals,
    )
    merged.update(
        _run_eval(
            runner,
            n,
            eval_llm_max_horizon=0,
            seed_base=seed_base,
            metric_prefix='eval_bot',
        )
    )
  merged['eval_full/elapsed_sec'] = time.time() - t0
  runner._backend.model.train()

  logging.info('--- Baseline evaluation (pass 0) ---')
  for k, v in sorted(merged.items()):
    logging.info('  %s: %.4f', k, v)
  try:
    with open(cache_path, 'w') as f:
      json.dump(
          {
              'episodes': n,
              'seed_base': seed_base,
              'bot_type': getattr(runner._config, 'bot_type', None),
              'metrics': merged,
              'timestamp': time.time(),
          },
          f,
          indent=2,
      )
  except IOError as e:
    logging.warning('[baselines] Failed to write %s: %s', cache_path, e)
  if runner._log_eval_metrics_fn is not None:
    # Zero budget counters anchor the pass-0 baseline at x=0 on the
    # decision-point dashboards (rows without an x value are not plotted).
    merged.update(_grpo_budget_metrics({}, {}))
    runner._log_eval_metrics_fn(0, merged)


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
      'constrained_actions=%s, strategic_actions=%s',
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
  )

  start_time = time.time()
  start_pass = getattr(runner._config, 'initial_pass', 1) or 1
  total_episodes_so_far = (start_pass - 1) * runner._config.collect_episodes
  # Cumulative sample-budget counters (see _GRPO_BUDGET_KEYS).  A resumed run
  # reads them back from eval_metrics.csv; they are not a function of the
  # pass index like total_episodes_so_far.
  budget_totals = {key: 0 for key in _GRPO_BUDGET_KEYS}
  if start_pass > 1:
    budget_totals = _load_grpo_budget_totals(
        os.path.join(runner._output_dir, 'results'), total_episodes_so_far
    )

  if start_pass > 1:
    logging.info(
        '=== [AUTO-RESUME] Resuming GRPO training from pass %d/%d (total_episodes=%d) ===',
        start_pass,
        runner._config.passes,
        total_episodes_so_far,
    )
    print(
        f'=== [AUTO-RESUME] Resuming GRPO training from pass {start_pass}/{runner._config.passes} '
        f'(episodes={total_episodes_so_far}) ===',
        flush=True,
    )

  if start_pass > runner._config.passes:
    logging.info(
        'Training already complete! start_pass (%d) > passes (%d). Exiting.',
        start_pass,
        runner._config.passes,
    )
    return

  # Fail fast on an unsupported eval_mode (before any pass is trained) and
  # record which per-pass evaluation flavour(s) this run will produce.
  _eval_mode = getattr(runner._config, 'eval_mode', 'both')
  _run_bot_guided, _run_full = _eval_flavours(_eval_mode)
  logging.info(
      '[evaluation] eval_mode=%r -> per-pass bot-guided curriculum eval'
      ' (eval/*, %d games): %s; full self-play eval (eval_full/*, %d games):'
      ' %s.',
      _eval_mode,
      runner._config.num_eval_episodes,
      'on' if _run_bot_guided else 'off',
      int(getattr(runner._config, 'eval_full_episodes', 0) or 0),
      'on' if _run_full else 'off',
  )

  # ── Pass 0: one-off reference baselines on the fixed eval deals ──
  # (initial adapter in full self-play + all-bot).  Only on a fresh start; a
  # resumed run finds results/eval_baselines.json and skips.
  if start_pass == 1:
    _maybe_run_baselines(runner, os.path.join(runner._output_dir, 'results'))

  for pass_idx in range(start_pass, runner._config.passes + 1):
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

    # ── Snapshot LoRA weights for stable partner simulation ──
    runner._frozen_lora_state = {
        name: param.data.clone()
        for name, param in runner._backend.model.named_parameters()
        if param.requires_grad
    }

    curriculum_ws = getattr(runner._config, 'curriculum_window_size', 0)
    is_graduated = getattr(runner, '_curriculum_graduated', False)
    if is_graduated:
      logging.info(
          '=== Pass %d/%d (Curriculum Graduated: Full Fine-Tuning across all turns) ===\n'
          '  Curriculum Window: Frozen (graduated to full-game fine-tuning)\n'
          '  Sampling:          All unique decision points across collected episodes\n'
          '  Collection & Eval: Full episodes (no horizon truncation/handover)',
          pass_idx,
          runner._config.passes,
      )
    elif curriculum_ws > 0:
      c_phase = (
          (pass_idx - 1) // max(1, runner._config.curriculum_passes_per_phase)
      ) + 1
      c_start, c_end = runner._config.get_curriculum_active_window(pass_idx)
      c_max_lb = getattr(runner._config, 'curriculum_max_lookback', -1)
      c_lb_start = max(0, c_start - c_max_lb) if c_max_lb >= 0 else 0
      c_max_h = getattr(runner._config, 'curriculum_max_horizon', 0)
      c_max_h_str = (
          f'{c_max_h}' if c_max_h > 0 else '0 (uncapped, train to end of game)'
      )
      logging.info(
          '=== Pass %d/%d (Curriculum Phase %d) ===\n'
          '  Curriculum Window: turns [%d, %d) (size %d, %d passes/phase)\n'
          '  Lookback Window:   turns [%d, %d) (max lookback: %s, replay'
          ' ratio: %.2f)\n'
          '  Max Horizon:       %s (collection horizon: turn %d)\n'
          '  Eval Handover:     turn %d (LLM plays [0, %d), bot plays [%d,'
          ' terminal))',
          pass_idx,
          runner._config.passes,
          c_phase,
          c_start,
          c_end,
          curriculum_ws,
          runner._config.curriculum_passes_per_phase,
          c_lb_start,
          c_start,
          f'{c_max_lb} turns' if c_max_lb >= 0 else 'unlimited [0, start_turn)',
          runner._config.curriculum_replay_ratio,
          c_max_h_str,
          c_end,
          c_end,
          c_end,
          c_end,
      )
    else:
      logging.info(
          '=== Pass %d/%d (No Curriculum: full episodes) ===',
          pass_idx,
          runner._config.passes,
      )

    # ── Step 1: Collect prompts (or load from cache if resuming mid-pass) ──
    results_dir = os.path.join(runner._output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    batch_cache_path = os.path.join(
        results_dir, f'batch_cache_pass{pass_idx}.json'
    )
    legacy_batch_cache_path = os.path.join(
        runner._output_dir, f'batch_cache_pass{pass_idx}.json'
    )
    interim_progress_path = os.path.join(results_dir, 'interim_progress.json')

    prompt_entries = None
    collect_stats = None
    sampled_player_prompts: dict[str, list[str]] = {}

    cache_to_read = (
        batch_cache_path
        if os.path.exists(batch_cache_path)
        else (
            legacy_batch_cache_path
            if os.path.exists(legacy_batch_cache_path)
            else None
        )
    )
    if cache_to_read is not None:
      try:
        with open(cache_to_read, 'r') as f:
          cached_payload = json.load(f)
        prompt_entries = cached_payload.get('prompt_entries')
        collect_stats = cached_payload.get('collect_stats')
        sampled_player_prompts = cached_payload.get(
            'sampled_player_prompts', {}
        )
        if prompt_entries and collect_stats:
          first_ser = prompt_entries[0].get('serialized_state')
          if (
              isinstance(first_ser, str)
              and '"adapter": "hanabi_env"' in first_ser
              and '"history"' not in first_ser
          ):
            logging.warning(
                '[BATCH CACHE] Legacy cache %s lacks Hanabi move history;'
                ' discarding and re-collecting pass %d.',
                cache_to_read,
                pass_idx,
            )
            prompt_entries = None
            sampled_player_prompts = {}
          else:
            for p_entry in prompt_entries:
              runner._prompt_metadata[p_entry['prompt']] = p_entry
            logging.info(
                '[BATCH CACHE] Restored %d cached rollout prompts for pass %d'
                ' from %s',
                len(prompt_entries),
                pass_idx,
                cache_to_read,
            )
            print(
                f'[BATCH CACHE] Restored {len(prompt_entries)} cached rollout'
                f' prompts for pass {pass_idx}!',
                flush=True,
            )
        else:
          prompt_entries = None
      except Exception as e:
        logging.warning(
            '[BATCH CACHE] Failed to load cache file %s: %s. Re-collecting.',
            cache_to_read,
            e,
        )
        prompt_entries = None

    resumed_completed_players: list[int] = []
    resumed_player_id = None
    resumed_completed_prompts = 0
    if prompt_entries is not None and os.path.exists(interim_progress_path):
      try:
        with open(interim_progress_path, 'r') as f:
          ip_data = json.load(f)
        if (
            ip_data.get('status') == 'in_progress'
            and ip_data.get('pass_idx') == pass_idx
        ):
          resumed_completed_players = [
              int(x) for x in ip_data.get('completed_players', [])
          ]
          resumed_player_id = ip_data.get('player_id')
          resumed_completed_prompts = int(
              ip_data.get('completed_prompts', 0)
          )
          logging.info(
              '[INTERIM RESUME] Pass %d resuming mid-pass:'
              ' completed_players=%s, current_player=%s,'
              ' completed_prompts=%d',
              pass_idx,
              resumed_completed_players,
              resumed_player_id,
              resumed_completed_prompts,
          )
      except Exception as e:
        logging.warning(
            '[INTERIM RESUME] Failed to read %s: %s', interim_progress_path, e
        )

    runner._pass_completed_players = list(resumed_completed_players)
    runner._pass_completed_prompts = 0
    if getattr(runner, '_last_periodic_save_time', 0.0) <= 0.0:
      runner._last_periodic_save_time = time.time()

    freshly_collected = False
    if prompt_entries is None:
      freshly_collected = True
      runner._backend.model.eval()
      prompt_entries, collect_stats = collect_game_prompts(
          runner,
          num_episodes=runner._config.collect_episodes,
          pass_idx=pass_idx,
          start_time=start_time,
      )
      try:
        with open(batch_cache_path, 'w') as f:
          json.dump(
              {
                  'prompt_entries': prompt_entries,
                  'collect_stats': collect_stats,
                  'sampled_player_prompts': sampled_player_prompts,
              },
              f,
          )
      except Exception as e:
        logging.warning('[BATCH CACHE] Failed to save cache: %s', e)

    total_episodes_so_far += runner._config.collect_episodes

    # Record collection metrics to results/collection_metrics.csv (only once per pass)
    if freshly_collected and os.path.exists(results_dir):
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

    # ── Step 2–3: Train (per-player or combined) with Curriculum Window Sampling ──
    total_pass_loss = 0.0
    total_pass_reward = 0.0
    num_train_steps = 0
    # Unique decision points expanded into GRPO groups this pass (summed over
    # per-player updates).  Counted from the full sampled list, before the
    # mid-pass resume skips, so a preempted pass is not under-counted.
    pass_decision_points = 0

    # Curriculum window calculation and graduation check:
    window_size = getattr(runner._config, 'curriculum_window_size', 0)
    if window_size > 0 and not getattr(runner, '_curriculum_graduated', False):
      start_turn, end_turn = runner._config.get_curriculum_active_window(
          pass_idx
      )
      max_horizon_cfg = getattr(runner._config, 'curriculum_max_horizon', 0)
      max_game_length = max(
          (e.get('turn_index', 0) + 1 for e in prompt_entries), default=0
      )
      active_count = sum(
          1
          for e in prompt_entries
          if start_turn <= e.get('turn_index', 0) < end_turn
      )
      reached_max_h = max_horizon_cfg > 0 and end_turn >= max_horizon_cfg
      reached_natural_end = max_horizon_cfg <= 0 and (
          start_turn >= max_game_length or active_count == 0
      )
      if reached_max_h or reached_natural_end:
        runner._curriculum_graduated = True
        logging.info(
            '[curriculum] Pass %d: Reached end of curriculum horizon '
            '(window=[%d, %d), max_turns_played=%d, active_prompts=%d). '
            'Graduating curriculum! Stopping horizon advancement and switching '
            'to full fine-tuning across all turns.',
            pass_idx,
            start_turn,
            end_turn,
            max_game_length,
            active_count,
        )

    is_graduated = getattr(runner, '_curriculum_graduated', False)
    if is_graduated or window_size <= 0:
      start_turn, end_turn = 0, 1000
      replay_ratio = 0.0
      lookback_start = 0
    else:
      start_turn, end_turn = runner._config.get_curriculum_active_window(
          pass_idx
      )
      replay_ratio = getattr(runner._config, 'curriculum_replay_ratio', 0.30)
      max_lookback = getattr(runner._config, 'curriculum_max_lookback', -1)
      lookback_start = (
          max(0, start_turn - max_lookback) if max_lookback >= 0 else 0
      )
      logging.info(
          '[curriculum] Pass %d: active window turns [%d, %d), lookback [%d,'
          ' %d) (max lookback: %s), replay ratio %.2f',
          pass_idx,
          start_turn,
          end_turn,
          lookback_start,
          start_turn,
          f'{max_lookback} turns' if max_lookback >= 0 else 'unlimited',
          replay_ratio,
      )

    def _sample_curriculum_prompts(entries: list[dict]) -> list[str]:
      """Selects unique prompts balancing active window and replay history."""
      if (
          getattr(runner, '_curriculum_graduated', False)
          or window_size <= 0
          or not entries
      ):
        all_unique = list({e['prompt'] for e in entries})
        logging.info(
            '[curriculum] Pass %d: Full fine-tuning mode. Optimizing all %d'
            ' unique prompts across all turns (%d total entries).',
            pass_idx,
            len(all_unique),
            len(entries),
        )
        return all_unique

      all_prompts = list({e['prompt'] for e in entries})
      max_game_length = max(
          (e.get('turn_index', 0) + 1 for e in entries), default=0
      )

      active_entries = [
          e for e in entries if start_turn <= e.get('turn_index', 0) < end_turn
      ]
      replay_entries = [
          e
          for e in entries
          if lookback_start <= e.get('turn_index', 0) < start_turn
      ]

      active_prompts = list({e['prompt'] for e in active_entries})
      replay_prompts = list({e['prompt'] for e in replay_entries})

      # Budget calculation:
      num_episodes = getattr(runner._config, 'collect_episodes', 1)
      if getattr(runner._config, 'per_player_updates', False):
        num_players = getattr(runner._env, 'num_players', 2)
        active_target = max(
            1, int(round((window_size * num_episodes) / num_players))
        )
      else:
        active_target = max(1, window_size * num_episodes)

      # No replay window (lookback 0, or start_turn == 0) => no replay budget;
      # otherwise the shortfall backfill below would refill it from any turn.
      clamped_ratio = (
          min(max(0.0, replay_ratio), 0.99) if lookback_start < start_turn else 0.0
      )
      max_updates = (
          int(round(active_target / (1.0 - clamped_ratio)))
          if clamped_ratio < 1.0
          else active_target * 2
      )
      lookback_target = max(0, max_updates - active_target)

      # ── Case 1: Active window completely falls out of collected episodes ──
      max_horizon_cfg = getattr(runner._config, 'curriculum_max_horizon', 0)
      is_completely_out = max_horizon_cfg <= 0 and (
          start_turn >= max_game_length or not active_prompts
      )
      if is_completely_out:
        runner._curriculum_graduated = True
        all_unique = list({e['prompt'] for e in entries})
        logging.info(
            '[curriculum] Pass %d: active window [%d, %d) completely out of'
            ' collected episodes (game max turn %d). Graduating curriculum to'
            ' full fine-tuning! Optimizing all %d unique prompts across all'
            ' turns.',
            pass_idx,
            start_turn,
            end_turn,
            max_game_length,
            len(all_unique),
        )
        return all_unique

      # ── Case 2: Active window partially or fully within collected episodes ──
      # 1. Sample from active window (up to active_target)
      n_active = min(len(active_prompts), active_target)
      selected_active = (
          list(np.random.choice(active_prompts, size=n_active, replace=False))
          if n_active > 0
          else []
      )

      # 2. Sample from lookback / replay window (up to lookback_target)
      if start_turn > 0 and replay_prompts and lookback_target > 0:
        avail_replay = list(set(replay_prompts) - set(selected_active))
        n_replay = min(len(avail_replay), lookback_target)
        selected_replay = (
            list(np.random.choice(avail_replay, size=n_replay, replace=False))
            if n_replay > 0
            else []
        )
      else:
        selected_replay = []

      selected_so_far = set(selected_active + selected_replay)

      # 3. If active window is partially out (or pool has shortfall),
      # distribute missed points back across any decision points in collected episodes uniformly.
      target_total = min(len(all_prompts), max_updates)
      shortfall = target_total - len(selected_so_far)
      extra_prompts = []
      if shortfall > 0:
        remaining_pool = list(set(all_prompts) - selected_so_far)
        n_extra = min(len(remaining_pool), shortfall)
        if n_extra > 0:
          extra_prompts = list(
              np.random.choice(remaining_pool, size=n_extra, replace=False)
          )

      selected = list(selected_so_far | set(extra_prompts))
      logging.info(
          '[curriculum] Sampled %d prompts (%d active [%d, %d), %d replay'
          ' [%d, %d), %d redistributed uniformly from missed turns; target'
          ' budget: %d)',
          len(selected),
          len(selected_active),
          start_turn,
          end_turn,
          len(selected_replay),
          lookback_start,
          start_turn,
          len(extra_prompts),
          max_updates,
      )
      return selected

    def _persist_batch_cache_with_prompts() -> None:
      try:
        with open(batch_cache_path, 'w') as f:
          json.dump(
              {
                  'prompt_entries': prompt_entries,
                  'collect_stats': collect_stats,
                  'sampled_player_prompts': sampled_player_prompts,
              },
              f,
          )
      except Exception as e:
        logging.warning('[BATCH CACHE] Failed to update cache: %s', e)

    if runner._config.per_player_updates:
      player_groups: dict[int, list[dict]] = {}
      for entry in prompt_entries:
        pid = entry['player_id']
        if pid not in player_groups:
          player_groups[pid] = []
        player_groups[pid].append(entry)

      for pid in sorted(player_groups.keys()):
        p_key = str(pid)
        if p_key in sampled_player_prompts and sampled_player_prompts[p_key]:
          unique_prompts = sampled_player_prompts[p_key]
        else:
          group_entries = player_groups[pid]
          unique_prompts = _sample_curriculum_prompts(group_entries)
          sampled_player_prompts[p_key] = unique_prompts
          _persist_batch_cache_with_prompts()

        if not unique_prompts:
          continue
        pass_decision_points += len(unique_prompts)
        if pid in runner._pass_completed_players:
          logging.info(
              '[INTERIM RESUME] Pass %d: Skipping Player %d (already completed'
              ' before preemption).',
              pass_idx,
              pid,
          )
          continue

        if (
            resumed_player_id == pid
            and 0 < resumed_completed_prompts < len(unique_prompts)
        ):
          logging.info(
              '[INTERIM RESUME] Pass %d (Player %d): Skipping %d/%d'
              ' already-trained prompts.',
              pass_idx,
              pid,
              resumed_completed_prompts,
              len(unique_prompts),
          )
          runner._pass_completed_prompts = resumed_completed_prompts
          prompts_to_train = unique_prompts[resumed_completed_prompts:]
        else:
          runner._pass_completed_prompts = 0
          prompts_to_train = unique_prompts

        logging.info(
            'Pass %d: training on %d unique prompts for Player %d.',
            pass_idx,
            len(prompts_to_train),
            pid,
        )
        p_loss, p_reward = _train_grpo_on_prompts(
            runner, prompts_to_train, pass_idx, pid, trl
        )
        total_pass_loss += p_loss
        total_pass_reward += p_reward
        num_train_steps += 1
        runner._pass_completed_players.append(pid)
        runner._pass_completed_prompts = 0
    else:
      p_key = 'all'
      if p_key in sampled_player_prompts and sampled_player_prompts[p_key]:
        unique_prompts = sampled_player_prompts[p_key]
      else:
        unique_prompts = _sample_curriculum_prompts(prompt_entries)
        sampled_player_prompts[p_key] = unique_prompts
        _persist_batch_cache_with_prompts()
      pass_decision_points += len(unique_prompts)

      if 0 < resumed_completed_prompts < len(unique_prompts):
        logging.info(
            '[INTERIM RESUME] Pass %d: Skipping %d/%d already-trained prompts.',
            pass_idx,
            resumed_completed_prompts,
            len(unique_prompts),
        )
        runner._pass_completed_prompts = resumed_completed_prompts
        prompts_to_train = unique_prompts[resumed_completed_prompts:]
      else:
        runner._pass_completed_prompts = 0
        prompts_to_train = unique_prompts

      logging.info(
          'Pass %d: %d unique prompts from %d total.',
          pass_idx,
          len(prompts_to_train),
          len(prompt_entries),
      )
      p_loss, p_reward = _train_grpo_on_prompts(
          runner, prompts_to_train, pass_idx, None, trl
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

    # Clean up mid-pass cache files now that training for this pass is complete
    try:
      for fname in os.listdir(results_dir):
        if fname.startswith('batch_cache_pass') or fname.startswith(
            'reward_cache_pass'
        ):
          os.remove(os.path.join(results_dir, fname))
    except Exception:
      pass

    # ── Step 4: Log training metrics ──
    if runner._log_training_step_fn is not None:
      runner._log_training_step_fn(pass_idx, avg_reward, avg_loss, start_time)

    # ── Step 5: Evaluate ──
    runner._backend.model.eval()
    eval_mode = getattr(runner._config, 'eval_mode', 'both')
    run_bot_guided, run_full = _eval_flavours(eval_mode)
    eval_metrics: dict[str, float] = {}
    if run_bot_guided:
      # Curriculum ("bot-guided") eval: the LLM plays turns [0, horizon) and
      # the heuristic bot finishes the game.  Only measures play up to the
      # curriculum horizon, so it is off unless eval_mode is bot_guided/both.
      horizon = (
          None
          if getattr(runner, '_curriculum_graduated', False)
          or runner._config.curriculum_window_size <= 0
          else runner._config.get_curriculum_horizon(pass_idx)
      )
      eval_kwargs = {}
      seed_base = _eval_seed_base(runner)
      if seed_base is not None:
        eval_kwargs['seed_base'] = seed_base
      if horizon is not None:
        eval_kwargs['eval_llm_max_horizon'] = horizon
        logging.info(
            '[curriculum evaluation] Pass %d: evaluating %d episodes. LLM'
            ' plays turns [0, %d); heuristic bot plays remainder'
            ' [%d, terminal). Deals: %s.',
            pass_idx,
            runner._config.num_eval_episodes,
            horizon,
            horizon,
            f'fixed (seed_base={seed_base})'
            if seed_base is not None
            else 'random',
        )
      else:
        logging.info(
            '[curriculum evaluation] Pass %d: evaluating %d episodes with LLM'
            ' for all turns (no curriculum horizon limit). Deals: %s.',
            pass_idx,
            runner._config.num_eval_episodes,
            f'fixed (seed_base={seed_base})'
            if seed_base is not None
            else 'random',
        )
      eval_metrics.update(
          _run_eval(runner, runner._config.num_eval_episodes, **eval_kwargs)
      )
    else:
      logging.info(
          '[curriculum evaluation] Pass %d: skipped (eval_mode=%r).',
          pass_idx,
          eval_mode,
      )
    if run_full:
      # Full self-play on the same fixed deals (LLM every turn): the
      # horizon-independent "is the whole game getting better?" signal.
      eval_metrics.update(_run_full_selfplay_eval(runner, pass_idx))
    eval_metrics['eval/collection_mean_reward'] = collect_stats['mean_reward']
    # Sample-budget counters: per pass and cumulative (x-axes of the
    # decision-point Flatboard dashboard, see _GRPO_BUDGET_KEYS).
    pass_budget = {
        'decision_points': pass_decision_points,
        'rollouts': (
            pass_decision_points
            * runner._config.num_generations
            * max(1, int(getattr(runner._config, 'train_epochs', 1) or 1))
        ),
        'collected_decision_points': int(collect_stats.get('num_prompts', 0)),
    }
    for key, count in pass_budget.items():
      budget_totals[key] += count
    eval_metrics.update(_grpo_budget_metrics(pass_budget, budget_totals))
    # Fold this pass's training diagnostics (mean over per-player calls) into
    # the eval row so loss decomposition / LoRA drift land in eval_metrics.csv.
    diags = getattr(runner, '_train_diags', [])
    for k in {k for d in diags for k in d}:
      eval_metrics[k] = float(np.mean([d[k] for d in diags if k in d]))
    runner._train_diags = []
    logging.info('--- Evaluation after GRPO pass %d ---', pass_idx)
    for k, v in sorted(eval_metrics.items()):
      logging.info('  %s: %.4f', k, v)

    if runner._log_eval_metrics_fn is not None:
      runner._log_eval_metrics_fn(total_episodes_so_far, eval_metrics)

    # ── Step 6: Checkpoint (runs asynchronously in parallel without blocking) ──
    try:
      with open(interim_progress_path, 'w') as f:
        json.dump(
            {
                'status': 'pass_complete',
                'pass_idx': pass_idx,
                'total_episodes': total_episodes_so_far,
                'timestamp': time.time(),
            },
            f,
            indent=2,
        )
    except Exception:
      pass

    checkpoint_meta = {
        'status': 'pass_complete',
        'pass_idx': pass_idx,
        'total_episodes': total_episodes_so_far,
        'eval_metrics': eval_metrics,
        'train_metrics': {
            'avg_loss': avg_loss,
            'avg_reward': avg_reward,
            'pass_elapsed_sec': pass_elapsed,
        },
        'collection_metrics': collect_stats,
        'timestamp': time.time(),
    }
    runner._last_periodic_save_time = time.time()
    runner._save_checkpoint_fn(total_episodes_so_far, metadata=checkpoint_meta)

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
  final_meta = {
      'pass_idx': runner._config.passes,
      'total_episodes': total_episodes_so_far,
      'total_time_sec': total_time,
      'timestamp': time.time(),
  }
  runner._save_checkpoint_fn(
      total_episodes_so_far, suffix='final', metadata=final_meta, wait=True
  )
