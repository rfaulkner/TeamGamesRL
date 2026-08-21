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

"""GRPO (Group Relative Policy Optimization) training for TeamGamesRL.

This module implements the GRPO training loop using TRL's GRPOTrainer.
GRPO collects game-state prompts by playing episodes, then trains the model
using group-relative advantage estimation over multiple completions per prompt.

The workflow:
  1. Play ``collect_episodes`` games to gather (prompt, action) pairs tied
     to specific game states.
  2. Define a reward function that simulates completing the game from each
     state to get outcome-based rewards.
  3. Feed prompts and the reward function to TRL's GRPOTrainer.
  4. Repeat for ``passes`` rounds, evaluating between rounds.

Usage:
  from learn.grpo import GRPOConfig, GRPORunner

  config = GRPOConfig(num_generations=4, passes=10)
  runner = GRPORunner(env, renderers, agents, backend, game_config,
                      evaluate_fn, save_checkpoint_fn, output_dir, config)
  runner.run()
"""

import dataclasses
import os
import time
from typing import Callable, Optional

from absl import logging
from learn.trajectory import PlayerTrajectory
from learn.trajectory import RLTrajectoryStep
import numpy as np
import torch


@dataclasses.dataclass
class GRPOConfig:
  """Configuration for the GRPO training algorithm.

  Attributes:
    num_generations: Number of completions to sample per prompt in GRPO (group
      size K).
    collect_episodes: Number of episodes to play for collecting game state
      prompts before each GRPO training pass.
    train_epochs: Number of training epochs per GRPO pass.
    passes: Number of collect-then-train passes for GRPO.
    max_completion_length: Maximum completion length for GRPO generation.
    lr: Learning rate for the optimizer.
    kl_coeff: KL penalty coefficient.
    max_grad_norm: Maximum gradient norm for clipping.
    temperature: Sampling temperature for LLM action selection during data
      collection.
    num_eval_episodes: Number of episodes per evaluation round.
    per_player_updates: If True, group collected prompts by player_id and run a
      separate GRPO training step for each player group. This gives cleaner
      gradient signal in cooperative games where players have different
      information (e.g. Tiny Hanabi).
    temperature_anneal_end: If set, linearly anneal the sampling temperature
      from ``temperature`` to this value over the course of training. None means
      no annealing (constant temperature).
    reward_variance_penalty: Coefficient for penalizing high-variance reward
      outcomes. The effective reward becomes ``mean_reward - penalty *
      std_reward`` when ``reward_num_simulations > 1``. Encourages convergence
      to consistent strategies.
    reward_num_simulations: Number of independent game simulations to run per
      completion for estimating reward mean and variance. Only used when
      ``reward_variance_penalty > 0``.
  """

  num_generations: int = 8
  collect_episodes: int = 50
  train_epochs: int = 1
  passes: int = 25
  max_completion_length: int = 16
  lr: float = 3e-5
  kl_coeff: float = 0.05
  max_grad_norm: float = 1.0
  temperature: float = 0.8
  num_eval_episodes: int = 10
  per_player_updates: bool = False
  temperature_anneal_end: float | None = None
  reward_variance_penalty: float = 0.0
  reward_num_simulations: int = 1


class GRPORunner:
  """Encapsulates the GRPO training loop using TRL's GRPOTrainer.

  This runner:
    1. Collects game-state prompts by playing episodes with LLM agents.
    2. Logs per-episode actions to console and episode_log.jsonl.
    3. Defines a reward function that simulates game completion from
       collected states using the appropriate state renderer.
    4. Trains the model using TRL's GRPOTrainer with group-relative
       advantage estimation.
    5. Evaluates and logs metrics to CSV after each pass.
    6. Saves checkpoints and writes a final summary JSON.

  Attributes:
    _env: The OpenSpiel rl_environment.Environment.
    _renderers: List of BaseStateRenderer, one per player.
    _agents: List of LLMAgent, one per player.
    _backend: The LLM backend (with model, tokenizer, generate_with_logprobs).
    _game_config: The GameConfig for the current game.
    _evaluate_fn: Callable (num_episodes) -> dict of eval metrics.
    _save_checkpoint_fn: Callable (episode, suffix) -> checkpoint path.
    _output_dir: Directory for logs and checkpoints.
    _config: The GRPOConfig.
    _prompt_metadata: Dict mapping prompt text to game-state metadata needed for
      reward simulation.
    _log_eval_metrics_fn: Optional callable (episode, metrics_dict) -> None.
    _log_training_step_fn: Optional callable (episode, reward, loss, start_time)
      -> None.
    _log_episode_fn: Optional callable (episode, trajectories, loss, is_eval) ->
      None.
    _write_summary_fn: Optional callable (total_time) -> None.
    _update_metrics_fn: Optional callable (trajectories, loss) -> float.
  """

  def __init__(
      self,
      env,
      renderers,
      agents,
      backend,
      game_config,
      evaluate_fn: Callable[[int], dict[str, float]],
      save_checkpoint_fn: Callable[..., str],
      output_dir: str,
      config: GRPOConfig,
      log_eval_metrics_fn: Optional[
          Callable[[int, dict[str, float]], None]
      ] = None,
      log_training_step_fn: Optional[
          Callable[[int, float, float, float], None]
      ] = None,
      log_episode_fn: Optional[
          Callable[[int, list[PlayerTrajectory], float, bool], None]
      ] = None,
      write_summary_fn: Optional[Callable[[float], None]] = None,
      update_metrics_fn: Optional[
          Callable[[list[PlayerTrajectory], float], float]
      ] = None,
  ):
    """Initializes the GRPORunner.

    Args:
      env: The OpenSpiel rl_environment.Environment.
      renderers: List of BaseStateRenderer, one per player.
      agents: List of LLMAgent, one per player.
      backend: The LLM backend (with model, tokenizer, generate_with_logprobs).
      game_config: The GameConfig for the current game.
      evaluate_fn: A callable ``(num_episodes: int) -> dict[str, float]`` for
        running evaluation.
      save_checkpoint_fn: A callable ``(episode: int, suffix: str) -> str`` for
        saving checkpoints.
      output_dir: The output directory path.
      config: A GRPOConfig instance.
      log_eval_metrics_fn: Optional callable for logging eval metrics to CSV.
      log_training_step_fn: Optional callable for logging training step metrics
        to CSV.
      log_episode_fn: Optional callable for writing JSONL episode transcripts.
      write_summary_fn: Optional callable for writing summary.json.
      update_metrics_fn: Optional callable for accumulating trainer metrics.
    """
    self._env = env
    self._renderers = renderers
    self._agents = agents
    self._backend = backend
    self._game_config = game_config
    self._evaluate_fn = evaluate_fn
    self._save_checkpoint_fn = save_checkpoint_fn
    self._output_dir = output_dir
    self._config = config
    self._log_eval_metrics_fn = log_eval_metrics_fn
    self._log_training_step_fn = log_training_step_fn
    self._log_episode_fn = log_episode_fn
    self._write_summary_fn = write_summary_fn
    self._update_metrics_fn = update_metrics_fn
    self._prompt_metadata: dict = {}
    self._current_temperature = config.temperature

  def collect_game_prompts(
      self, num_episodes: int, pass_idx: int = 1, start_time: float = 0.0
  ) -> list[dict]:
    """Collects game-state prompts by playing episodes with LLM agents.

    Plays ``num_episodes`` games and records the prompts shown to each
    player at each decision point, along with the action history and
    player index needed to simulate game completion for reward
    computation.

    Args:
      num_episodes: Number of episodes to play for prompt collection.
      pass_idx: The current GRPO pass index.
      start_time: Training start timestamp for elapsed time calculation.

    Returns:
      A list of dicts, each containing:
        - 'prompt': The full LLM prompt string.
        - 'player_id': The player who saw this prompt.
        - 'action_history': List of action IDs taken so far in the game.
        - 'legal_actions': List of legal action IDs at this state.
        - 'legal_actions_desc': List of (action_id, desc) tuples.
        - 'state_text': Rendered state text.
    """
    all_prompts = []
    num_players = self._game_config.num_players
    ep_rewards = []

    for ep in range(1, num_episodes + 1):
      time_step = self._env.reset()
      action_history = []
      trajectories = [PlayerTrajectory(player_id=p) for p in range(num_players)]

      while not time_step.last():
        current_player = time_step.current_player()
        state = self._env._state  # pylint: disable=protected-access

        # Render state and legal actions.
        state_text = self._renderers[current_player].render_state(
            state, current_player, self._env.game
        )
        legal_actions_with_desc = self._renderers[
            current_player
        ].render_legal_actions(state, current_player, self._env.game)
        legal_actions = [a for a, _ in legal_actions_with_desc]
        action_descriptions = [d for _, d in legal_actions_with_desc]

        # Build prompt.
        prompt = self._agents[current_player]._build_prompt(  # pylint: disable=protected-access
            state_text, legal_actions, action_descriptions
        )

        # Select action using the LLM.
        response, log_prob = self._backend.generate_with_logprobs(
            prompt,
            temperature=self._current_temperature,
            max_tokens=self._config.max_completion_length,
        )
        action_id = self._renderers[current_player].parse_action(
            response, legal_actions_with_desc
        )
        if action_id is None:
          action_id = int(np.random.choice(legal_actions))

        action_text = state.action_to_string(current_player, action_id)

        prompt_entry = {
            'prompt': prompt,
            'player_id': current_player,
            'action_history': list(action_history),
            'legal_actions': legal_actions,
            'legal_actions_desc': legal_actions_with_desc,
            'state_text': state_text,
        }
        all_prompts.append(prompt_entry)
        self._prompt_metadata[prompt] = prompt_entry

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
        time_step = self._env.step([action_id])

      # Assign rewards.
      if time_step.rewards is not None:
        for p in range(num_players):
          trajectories[p].reward = time_step.rewards[p]

      mean_r = float(np.mean([t.reward for t in trajectories]))
      ep_rewards.append(mean_r)

      # Log episode transcript to file if logger available.
      global_ep = (pass_idx - 1) * num_episodes + ep
      if self._log_episode_fn is not None:
        self._log_episode_fn(global_ep, trajectories, 0.0, False)

      if self._update_metrics_fn is not None:
        self._update_metrics_fn(trajectories, 0.0)

      # Print concise progress to stdout so actions are clearly visible in logs.
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
        'Pass %d collection complete: %d prompts from %d episodes (mean reward:'
        ' %.3f)',
        pass_idx,
        len(all_prompts),
        num_episodes,
        mean_collected,
    )
    return all_prompts

  def _simulate_from_state(
      self,
      action_history: list[int],
      chosen_action: int,
      target_player: int,
  ) -> float:
    """Simulates a game to completion from a given state to get the reward.

    Replays ``action_history`` to reconstruct the game state, applies
    ``chosen_action``, then plays out the rest of the game with random
    actions to estimate the reward for ``target_player``.

    Args:
      action_history: List of action IDs taken before the current decision.
      chosen_action: The action to apply at the current decision point.
      target_player: The player whose reward we want.

    Returns:
      The reward for ``target_player`` at the end of the simulated game.
    """
    time_step = self._env.reset()

    # Replay action history.
    for action_id in action_history:
      if time_step.last():
        break
      time_step = self._env.step([action_id])

    # Apply the chosen action.
    if not time_step.last():
      time_step = self._env.step([chosen_action])

    # Play out the rest of the game with the current policy (self-play).
    while not time_step.last():
      current_player = time_step.current_player()
      state = self._env._state  # pylint: disable=protected-access

      state_text = self._renderers[current_player].render_state(
          state, current_player, self._env.game
      )
      legal_actions_with_desc = self._renderers[
          current_player
      ].render_legal_actions(state, current_player, self._env.game)
      legal_actions = [a for a, _ in legal_actions_with_desc]
      action_descriptions = [d for _, d in legal_actions_with_desc]

      prompt = self._agents[current_player]._build_prompt(  # pylint: disable=protected-access
          state_text, legal_actions, action_descriptions
      )

      with torch.no_grad():
        response, _ = self._backend.generate_with_logprobs(
            prompt,
            temperature=self._current_temperature,
            max_tokens=self._config.max_completion_length,
        )

      partner_action = self._renderers[current_player].parse_action(
          response, legal_actions_with_desc
      )
      if partner_action is None:
        partner_action = (
            int(np.random.choice(legal_actions)) if legal_actions else 0
        )

      time_step = self._env.step([partner_action])

    if time_step.rewards is not None:
      return float(time_step.rewards[target_player])
    return 0.0

  def run(self) -> None:
    """Runs the full GRPO training loop.

    For each pass:
      1. Collect game-state prompts by playing episodes (logging actions).
      2. Build a reward function using state renderers and simulation.
      3. Create a TRL GRPOTrainer and train for one epoch.
      4. Log training metrics to CSV.
      5. Evaluate the updated policy and log eval metrics to CSV.
      6. Save a checkpoint.
    """
    from backend.gemma_backend import _lazy_import_hf  # pylint: disable=g-import-not-at-top

    _lazy_import_hf()
    import trl  # pylint: disable=g-import-not-at-top

    logging.info(
        'Starting GRPO training: %d passes, %d episodes/pass, K=%d',
        self._config.passes,
        self._config.collect_episodes,
        self._config.num_generations,
    )

    start_time = time.time()
    total_episodes_so_far = 0

    for pass_idx in range(1, self._config.passes + 1):
      pass_start = time.time()
      logging.info('=== GRPO pass %d/%d ===', pass_idx, self._config.passes)

      # ── Temperature annealing ──
      if self._config.temperature_anneal_end is not None:
        progress = (pass_idx - 1) / max(self._config.passes - 1, 1)
        self._current_temperature = self._config.temperature + progress * (
            self._config.temperature_anneal_end - self._config.temperature
        )
        logging.info(
            'Temperature annealed to %.3f (pass %d/%d)',
            self._current_temperature,
            pass_idx,
            self._config.passes,
        )

      # ── Step 1: Collect prompts ──
      self._backend.model.eval()
      prompt_entries = self.collect_game_prompts(
          num_episodes=self._config.collect_episodes,
          pass_idx=pass_idx,
          start_time=start_time,
      )
      total_episodes_so_far += self._config.collect_episodes

      if not prompt_entries:
        logging.warning('No prompts collected in pass %d, skipping.', pass_idx)
        continue

      # ── Step 2–3: Train (per-player or combined) ──
      if self._config.per_player_updates:
        # Group prompts by player for separate GRPO updates.
        # Each player's decision points get their own training step so that
        # GRPO's group-relative advantage is computed within the context of
        # a single player's information set (e.g. P0 acts without seeing
        # P1's card, while P1 sees P0's action).
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
          self._train_grpo_on_prompts(unique_prompts, pass_idx, pid, trl)
      else:
        # Original behavior: all prompts together.
        unique_prompts = list({e['prompt'] for e in prompt_entries})
        logging.info(
            'Pass %d: %d unique prompts from %d total.',
            pass_idx,
            len(unique_prompts),
            len(prompt_entries),
        )
        self._train_grpo_on_prompts(unique_prompts, pass_idx, None, trl)

      pass_elapsed = time.time() - pass_start
      logging.info(
          'GRPO pass %d complete in %.1f sec.',
          pass_idx,
          pass_elapsed,
      )

      # ── Step 4: Log training metrics to CSV ──
      if self._log_training_step_fn is not None:
        self._log_training_step_fn(total_episodes_so_far, 0.0, 0.0, start_time)

      # ── Step 5: Evaluate & log eval metrics to CSV ──
      self._backend.model.eval()
      eval_metrics = self._evaluate_fn(self._config.num_eval_episodes)
      logging.info('--- Evaluation after GRPO pass %d ---', pass_idx)
      for k, v in sorted(eval_metrics.items()):
        logging.info('  %s: %.4f', k, v)

      if self._log_eval_metrics_fn is not None:
        self._log_eval_metrics_fn(total_episodes_so_far, eval_metrics)

      # ── Step 6: Checkpoint ──
      self._save_checkpoint_fn(total_episodes_so_far)

    # ── Final summary and checkpoint ──
    total_time = time.time() - start_time
    logging.info(
        'GRPO training complete: %d passes (%d episodes) in %.1f sec.',
        self._config.passes,
        total_episodes_so_far,
        total_time,
    )
    if self._write_summary_fn is not None:
      self._write_summary_fn(total_time)
    self._save_checkpoint_fn(total_episodes_so_far, suffix='final')

  def _train_grpo_on_prompts(
      self,
      unique_prompts: list[str],
      pass_idx: int,
      player_id: int | None,
      trl_module,
  ) -> tuple[float, float]:
    """Runs one GRPO training step on a set of prompts.

    Args:
      unique_prompts: Deduplicated prompt strings to train on.
      pass_idx: Current pass index (for output directory naming).
      player_id: If not None, only prompts for this player are included (used
        for per-player updates). Affects output dir naming.
      trl_module: The imported trl module.

    Returns:
      Tuple of (mean_loss, mean_reward) from the training step.
    """
    eval_counter = [0]

    def reward_fn(completions, prompts=None, **kwargs):
      """Compute rewards for GRPO completions via game simulation."""
      del kwargs  # Unused.
      rewards = []
      for i, completion in enumerate(completions):
        prompt_text = (
            prompts[i] if prompts is not None and i < len(prompts) else ''
        )
        metadata = self._prompt_metadata.get(prompt_text, {})
        action_history = metadata.get('action_history', [])
        p_id = metadata.get('player_id', 0)
        legal_actions = metadata.get('legal_actions', [])
        legal_actions_desc = metadata.get('legal_actions_desc', [])

        # Extract completion text.
        if hasattr(completion, 'text'):
          comp_text = completion.text
        elif isinstance(completion, list):
          comp_text = self._backend.tokenizer.decode(
              completion, skip_special_tokens=True
          )
        else:
          comp_text = str(completion)

        # Parse completion into an action ID using the renderer.
        action_id = self._renderers[p_id].parse_action(
            comp_text, legal_actions_desc
        )
        parsed = True
        if action_id is None:
          parsed = False
          if legal_actions:
            action_id = int(np.random.choice(legal_actions))
          else:
            action_id = 0

        # Reward computation with optional variance penalty.
        if (
            self._config.reward_num_simulations > 1
            and self._config.reward_variance_penalty > 0
        ):
          sim_rewards = [
              self._simulate_from_state(action_history, action_id, p_id)
              for _ in range(self._config.reward_num_simulations)
          ]
          mean_r = float(np.mean(sim_rewards))
          std_r = float(np.std(sim_rewards))
          reward = mean_r - self._config.reward_variance_penalty * std_r
        else:
          reward = self._simulate_from_state(action_history, action_id, p_id)
        rewards.append(torch.tensor(float(reward)))

        # Periodically log sample completions.
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

      return rewards

    # Build output directory.
    if player_id is not None:
      out_dir = os.path.join(
          self._output_dir, f'grpo_pass_{pass_idx}_p{player_id}'
      )
    else:
      out_dir = os.path.join(self._output_dir, f'grpo_pass_{pass_idx}')

    self._backend.model.train()
    # TRL GRPO requires:
    # 1. generation_batch_size % num_generations == 0
    # 2. generation_batch_size % (
    #       per_device_train_batch_size * num_processes) == 0
    # By setting generation_batch_size = num_generations and picking
    # per_device_train_batch_size as the largest divisor of
    #       num_generations <= 4,
    # both constraints are always satisfied regardless of len(unique_prompts).
    max_train_batch = 4
    candidates = [
        d
        for d in range(1, max_train_batch + 1)
        if self._config.num_generations % d == 0
    ]
    batch_size = max(candidates) if candidates else 1
    gen_batch_size = self._config.num_generations
    training_args = trl_module.GRPOConfig(
        output_dir=out_dir,
        num_train_epochs=self._config.train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=self._config.lr,
        max_grad_norm=self._config.max_grad_norm,
        logging_steps=1,
        save_strategy='no',
        max_completion_length=self._config.max_completion_length,
        num_generations=self._config.num_generations,
        generation_batch_size=gen_batch_size,
        beta=self._config.kl_coeff,
        temperature=self._current_temperature,
        report_to='none',
    )

    trainer = trl_module.GRPOTrainer(
        model=self._backend.model,
        args=training_args,
        reward_funcs=reward_fn,
        processing_class=self._backend.tokenizer,
        train_dataset=self._build_prompt_dataset(unique_prompts),
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

  def _build_prompt_dataset(self, prompts: list[str]):
    """Builds a HuggingFace Dataset from a list of prompt strings.

    Args:
      prompts: List of prompt strings.

    Returns:
      A datasets.Dataset with a single 'prompt' column.
    """
    from datasets import Dataset  # pylint: disable=g-import-not-at-top

    return Dataset.from_dict({'prompt': prompts})
