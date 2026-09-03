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

import copy
import os
import time

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
      if (runner._current_epsilon > 0
          and np.random.random() < runner._current_epsilon):
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
              prompt=prompt,
              action_text=response.strip(),
              action_id=action_id,
              log_prob=log_prob,
              state_text=state_text,
              llm_response=response,
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
    print(
        f'[pass {pass_idx} collect {ep}/{num_episodes}] reward={mean_r:.2f} '
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
  return all_prompts


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
# TRL-based GRPO training step
# ═══════════════════════════════════════════════════════════════════════


def _build_prompt_dataset(prompts: list[str]):
  """Build a HuggingFace Dataset from prompt strings."""
  from datasets import Dataset  # pylint: disable=g-import-not-at-top

  return Dataset.from_dict({'prompt': prompts})


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

  def reward_fn(completions, prompts=None, **kwargs):
    del kwargs
    rewards = []
    reward_cache = {}  # (prompt_text, action_id) -> reward tensor
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

      # Check reward cache for duplicate (prompt, action) pairs.
      cache_key = (prompt_text, action_id)
      if cache_key in reward_cache:
        rewards.append(reward_cache[cache_key])
        cache_hits += 1
        eval_counter[0] += 1
        if eval_counter[0] <= 5 or eval_counter[0] % 25 == 0:
          status = 'parsed' if parsed else 'random_fallback'
          logging.info(
              '[GRPO eval #%d] P%d | completion=%r '
              '-> action=%s (%s) | reward=%.1f (cached)',
              eval_counter[0],
              p_id,
              comp_text.strip()[:60],
              action_id,
              status,
              float(reward_cache[cache_key]),
          )
        continue

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
          reward = evaluate_action_quality(
              eval_state, action_id, p_id
          )

      elif sim_mode == 'dense_chain':
        # Dense rewards over a short heuristic continuation.
        from learn.action_reward import evaluate_dense_chain  # pylint: disable=g-import-not-at-top
        if not parsed:
          reward = -0.3  # Parse failure penalty.
        else:
          horizon = runner._config.truncated_rollout_horizon or 4
          discount = getattr(
              runner._config, 'dense_chain_discount', 0.9
          )
          reward = evaluate_dense_chain(
              runner, action_history, action_id, p_id,
              serialized_state=ser_state,
              horizon=horizon,
              discount=discount,
          )

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
      if eval_counter[0] <= 5 or eval_counter[0] % 25 == 0:
        status = 'parsed' if parsed else 'random_fallback'
        logging.info(
            '[GRPO eval #%d] P%d | completion=%r '
            '-> action=%s (%s) | reward=%.1f',
            eval_counter[0],
            p_id,
            comp_text.strip()[:60],
            action_id,
            status,
            reward,
        )

    if cache_hits > 0:
      logging.info(
          'Reward cache: %d/%d hits (%.0f%% duplicates avoided)',
          cache_hits,
          len(rewards),
          100.0 * cache_hits / len(rewards),
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

  max_train_batch = 4
  candidates = [
      d
      for d in range(1, max_train_batch + 1)
      if runner._config.num_generations % d == 0
  ]
  batch_size = max(candidates) if candidates else 1
  gen_batch_size = runner._config.num_generations

  training_args = trl_module.GRPOConfig(
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
  trainer.train()

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
      'reward_sim=%s, horizon=%s',
      runner._config.passes,
      runner._config.collect_episodes,
      runner._config.num_generations,
      runner._config.reward_simulation_mode,
      runner._config.truncated_rollout_horizon,
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
      runner._current_temperature = runner._config.temperature + progress * (
          runner._config.temperature_anneal_end - runner._config.temperature
      )
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
    prompt_entries = collect_game_prompts(
        runner,
        num_episodes=runner._config.collect_episodes,
        pass_idx=pass_idx,
        start_time=start_time,
    )
    total_episodes_so_far += runner._config.collect_episodes

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
