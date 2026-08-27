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

import copy
import dataclasses
import os
import time
from typing import Callable, Optional

from absl import logging
from learn.trajectory import PlayerTrajectory
from learn.trajectory import RLTrajectoryStep
import numpy as np
import pyspiel
import torch


@dataclasses.dataclass
class GRPOConfig:
  """Configuration for the GRPO training algorithm.

  Attributes:
    num_generations: Number of completions to sample per prompt in GRPO (group
      size K). Only used when ``exhaustive_groups`` is False.
    collect_episodes: Number of episodes to play for collecting game state
      prompts before each GRPO training pass. Only used when
      ``exhaustive_groups`` is False.
    train_epochs: Number of training epochs per GRPO pass.
    passes: Number of collect-then-train passes for GRPO.
    max_completion_length: Maximum completion length for GRPO generation.
    lr: Learning rate for the optimizer.
    kl_coeff: KL penalty coefficient.
    max_grad_norm: Maximum gradient norm for clipping.
    gradient_accumulation_steps: Number of groups to accumulate gradients over
      before taking an optimizer step. Defaults to 1 (step after every group).
    temperature: Sampling temperature for LLM action selection during data
      collection.
    num_eval_episodes: Number of episodes per evaluation round.
    per_player_updates: If True, group collected prompts by player_id and run a
      separate GRPO training step for each player group. This gives cleaner
      gradient signal in cooperative games where players have different
      information (e.g. Tiny Hanabi). Only used when ``exhaustive_groups`` is
      False.
    temperature_anneal_end: If set, linearly anneal the sampling temperature
      from ``temperature`` to this value over the course of training. None means
      no annealing (constant temperature).
    reward_variance_penalty: Coefficient for penalizing high-variance reward
      outcomes. The effective reward becomes ``mean_reward - penalty *
      std_reward`` when ``reward_num_simulations > 1``. Encourages convergence
      to consistent strategies. Only used when ``exhaustive_groups`` is False.
    reward_num_simulations: Number of independent game simulations to run per
      completion for estimating reward mean and variance. Only used when
      ``reward_variance_penalty > 0``.
    exhaustive_groups: If True, enumerate all possible game states and form
      GRPO groups where only the target player's action varies while all other
      variables (chance outcomes, other players' actions) are fixed. This
      produces deterministic, zero-variance advantage estimates by reading
      rewards directly from the game's payoff structure. For small games like
      Tiny Hanabi this covers the full game tree; for larger games, set this
      to False and use sampled rollouts instead.
    optimistic_reward_alpha: Blending weight for optimistic (max-over-partner)
      rewards in exhaustive-group GRPO.  For the first-acting player, rewards
      are computed as::

        reward = α * max_over_partner_actions(r) + (1 - α) * simulated(r)

      When α=1.0 (fully optimistic), the reward assumes the best possible
      partner cooperation, breaking the chicken-and-egg bootstrapping problem
      in signaling games.  When α=0.0, rewards reflect the partner's current
      policy.  The value is linearly annealed from ``optimistic_reward_alpha``
      toward ``optimistic_reward_alpha_min`` over training passes.
      Only used when ``exhaustive_groups`` is True.  Set to 0.0 to disable.
    optimistic_reward_alpha_min: Minimum floor for optimistic reward alpha
      annealing.  Maintains a baseline optimistic drive to prevent premature
      collapse into non-cooperative equilibria (such as the 8.0 fallback in
      Tiny Hanabi).  Defaults to 0.2.
  """

  num_generations: int = 8
  collect_episodes: int = 50
  train_epochs: int = 1
  passes: int = 25
  max_completion_length: int = 16
  lr: float = 3e-5
  kl_coeff: float = 0.05
  max_grad_norm: float = 1.0
  gradient_accumulation_steps: int = 1
  temperature: float = 0.8
  num_eval_episodes: int = 10
  per_player_updates: bool = True
  temperature_anneal_end: float | None = None
  reward_variance_penalty: float = 0.0
  reward_num_simulations: int = 5
  exhaustive_groups: bool = False
  optimistic_reward_alpha: float = 1.0
  optimistic_reward_alpha_min: float = 0.2
  signal_entropy_coeff: float = 0.0
  """Coefficient for the cross-state signal entropy bonus.

  In signaling games, the first-acting player (P0) may converge to the
  same action for all dealt cards, collapsing into a non-signaling
  equilibrium (e.g. the 8.0 plateau in Tiny Hanabi).

  When ``signal_entropy_coeff > 0``, an additional loss term is added
  after each GRPO pass that maximizes the entropy of P0's *marginal*
  action distribution across all game states::

    π_marginal(a) = (1/N) Σ_s π(a | s)
    L_entropy = -signal_entropy_coeff * H(π_marginal)

  This encourages P0 to use different actions for different states,
  breaking symmetry without prescribing a specific signaling convention.

  Only used when ``exhaustive_groups`` is True.  A value of 0.1–0.5 is
  recommended.  The bonus applies only to player 0 (the signaler).
  """
  phased_training: bool = False
  """Enable phased (curriculum) training with per-player LoRA adapters.

  When True, training proceeds in phases:
    - Phase 1: Train P1 only against oracle-best P0 actions.
    - Phase 2: Freeze P1 and train P0 against P1's learned policy.
    - Phase 3 (optional): Joint fine-tuning with both adapters.

  Each player gets its own LoRA adapter so gradients don't interfere.
  Only used when ``exhaustive_groups`` is True.
  """
  phase1_max_passes: int = 50
  """Maximum passes for Phase 1 (P1 training against oracle P0)."""
  phase2_max_passes: int = 50
  """Maximum passes for Phase 2 (P0 training against frozen P1)."""
  phase3_max_passes: int = 10
  """Maximum passes for Phase 3 (joint fine-tuning). Set to 0 to skip."""
  convergence_patience: int = 5
  """Stop a phase early if eval reward hasn't improved by more than
  ``convergence_min_delta`` for this many consecutive passes."""
  convergence_min_delta: float = 0.1
  """Minimum reward improvement to reset the patience counter."""


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
      ref_state_dict: dict | None = None,
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
    self._frozen_lora_state: dict | None = None
    self._ref_state_dict = ref_state_dict

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
        - 'serialized_state': Serialized game+state string for exact
            restoration during reward simulation.
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
            'serialized_state': pyspiel.serialize_game_and_state(
                self._env.game, state
            ),
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
      serialized_state: str | None = None,
  ) -> float:
    """Simulates a game to completion from a given state to get the reward.

    Restores the exact game state (preserving the original card deal) via
    ``serialized_state``, applies ``chosen_action``, then plays out the
    partner's remaining turns using frozen LoRA weights from the start of
    the current pass. This provides a stable training target so that the
    signaler can learn conventions the responder will decode consistently.

    Args:
      action_history: List of action IDs taken before the current decision.
      chosen_action: The action to apply at the current decision point.
      target_player: The player whose reward we want.
      serialized_state: Serialized game-and-state string from
        ``pyspiel.serialize_game_and_state()``. If provided, the exact
        game state (including the card deal) is restored. Falls back to
        action-replay on ``self._env`` if ``None``.

    Returns:
      The reward for ``target_player`` at the end of the simulated game.
    """
    # ── Restore the exact game state ──
    if serialized_state is not None:
      # Extract the state portion from the serialized game+state string.
      # Format: "<game_string>\n<state_string>", separated by a blank line.
      # We use the env's own game object so set_state()'s identity check passes.
      _, state = pyspiel.deserialize_game_and_state(serialized_state)
      # Re-create state using the env's game to pass identity check.
      restored = self._env.game.deserialize_state(state.serialize())
      self._env.set_state(restored)
    else:
      # Legacy fallback: replay from a fresh reset (re-deals cards).
      self._env.reset()
      state = self._env._state  # pylint: disable=protected-access
      for action_id in action_history:
        if state.is_terminal():
          break
        state.apply_action(action_id)

    # Apply the chosen action.
    state = self._env._state  # pylint: disable=protected-access
    if not state.is_terminal():
      state.apply_action(chosen_action)

    # ── Swap in frozen LoRA weights for partner simulation ──
    live_lora_state = None
    if self._frozen_lora_state is not None:
      live_lora_state = copy.deepcopy(
          {k: v for k, v in self._backend.model.named_parameters()
           if v.requires_grad}
      )
      for name, param in self._backend.model.named_parameters():
        if name in self._frozen_lora_state:
          param.data.copy_(self._frozen_lora_state[name])

    try:
      # Play out the rest of the game with the frozen partner policy.
      while not state.is_terminal():
        current_player = state.current_player()

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

        state.apply_action(partner_action)
    finally:
      # ── Restore live LoRA weights ──
      if live_lora_state is not None:
        for name, param in self._backend.model.named_parameters():
          if name in live_lora_state:
            param.data.copy_(live_lora_state[name].data)

    if state.is_terminal() and state.rewards() is not None:
      return float(state.rewards()[target_player])
    return 0.0

  def run(self) -> None:
    """Runs the full GRPO training loop.

    Dispatches to exhaustive-group GRPO when ``config.exhaustive_groups`` is
    True, otherwise uses the TRL-based sampled approach.

    For each pass:
      1. Collect game-state prompts by playing episodes (logging actions).
      2. Build a reward function using state renderers and simulation.
      3. Create a TRL GRPOTrainer and train for one epoch.
      4. Log training metrics to CSV.
      5. Evaluate the updated policy and log eval metrics to CSV.
      6. Save a checkpoint.
    """
    if self._config.exhaustive_groups and self._config.phased_training:
      self._run_phased_exhaustive()
      return

    if self._config.exhaustive_groups:
      self._run_exhaustive()
      return

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

      # ── Snapshot LoRA weights for stable partner simulation ──
      # Freeze a copy of the trainable parameters at the start of each pass
      # so that reward simulation uses a consistent partner policy.
      self._frozen_lora_state = {
          name: param.data.clone()
          for name, param in self._backend.model.named_parameters()
          if param.requires_grad
      }

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
      total_pass_loss = 0.0
      total_pass_reward = 0.0
      num_train_steps = 0

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
          p_loss, p_reward = self._train_grpo_on_prompts(
              unique_prompts, pass_idx, pid, trl
          )
          total_pass_loss += p_loss
          total_pass_reward += p_reward
          num_train_steps += 1
      else:
        # Original behavior: all prompts together.
        unique_prompts = list({e['prompt'] for e in prompt_entries})
        logging.info(
            'Pass %d: %d unique prompts from %d total.',
            pass_idx,
            len(unique_prompts),
            len(prompt_entries),
        )
        p_loss, p_reward = self._train_grpo_on_prompts(
            unique_prompts, pass_idx, None, trl
        )
        total_pass_loss += p_loss
        total_pass_reward += p_reward
        num_train_steps += 1

      pass_elapsed = time.time() - pass_start
      avg_pass_loss = (
          total_pass_loss / num_train_steps if num_train_steps else 0.0
      )
      avg_pass_reward = (
          total_pass_reward / num_train_steps if num_train_steps else 0.0
      )
      logging.info(
          'GRPO pass %d complete in %.1f sec (loss=%.4f, reward=%.2f).',
          pass_idx,
          pass_elapsed,
          avg_pass_loss,
          avg_pass_reward,
      )

      # ── Step 4: Log training metrics to CSV ──
      if self._log_training_step_fn is not None:
        self._log_training_step_fn(
            pass_idx, avg_pass_reward, avg_pass_loss, start_time
        )

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

  # ════════════════════════════════════════════════════════════════════════
  # Exhaustive-group GRPO
  # ════════════════════════════════════════════════════════════════════════

  def _enumerate_grpo_groups(
      self,
      optimistic_alpha: float = 0.0,
  ) -> list[dict[str, object]]:
    """Enumerate all GRPO groups by walking the game tree.

    A *group* is a set of (prompt, action, reward) tuples where all game
    variables — chance outcomes and other players' actions — are held fixed
    and only the *target player's action* varies.  This gives the cleanest
    possible GRPO advantage signal: zero variance, no cross-context noise.

    The method walks the game tree depth-first, branching at chance nodes
    and at the target-player's decision node.  At every other player's
    decision node, it queries the current LLM policy to select a single
    action (no branching), so the resulting groups reflect the *current*
    partner strategy.

    For small games like Tiny Hanabi (36 terminal states) this is an
    exhaustive walk.  For larger games the same structure works but the
    caller should set ``collect_episodes`` to limit the number of sampled
    rollout contexts.

    Returns:
      A list of group dicts, each containing:
        - ``'player_id'``: int — the player whose action varies.
        - ``'prompt'``:     str — the LLM prompt for the target player.
        - ``'actions'``:    list[int] — the legal action IDs.
        - ``'action_texts'``: list[str] — the action text for each action
            (what the LLM should output to select that action).
        - ``'rewards'``:    list[float] — the reward for each action.
        - ``'context'``:    str — a human-readable description of the
            fixed context (for logging).
    """
    game = self._env.game
    groups: list[dict[str, object]] = []

    def _walk(state, context_parts: list[str], target_player: int | None):
      """Recursively walk the game tree to build groups.

      Args:
        state: Current OpenSpiel game state.
        context_parts: Human-readable list describing the fixed context.
        target_player: Once set, this is the player whose decision node
          will be expanded into a group.  None means we haven't committed
          to a target player yet — the first player decision encountered
          on each path will branch into a group.
      """
      if state.is_terminal():
        return

      if state.is_chance_node():
        # Branch over all chance outcomes (e.g. card deals).
        for chance_action, _ in state.chance_outcomes():
          child = state.child(chance_action)
          action_str = state.action_to_string(
              pyspiel.PlayerId.CHANCE, chance_action
          )
          _walk(
              child,
              context_parts + [f'chance:{action_str}'],
              target_player,
          )
        return

      current_player = state.current_player()

      if target_player is None or current_player == target_player:
        # This is a decision point we want to expand into a GRPO group.
        # Render the prompt from this player's perspective.
        state_text = self._renderers[current_player].render_state(
            state, current_player, game
        )
        legal_actions_with_desc = self._renderers[
            current_player
        ].render_legal_actions(state, current_player, game)
        legal_ids = [a for a, _ in legal_actions_with_desc]
        action_descs = [d for _, d in legal_actions_with_desc]

        prompt = self._agents[current_player]._build_prompt(  # pylint: disable=protected-access
            state_text, legal_ids, action_descs
        )

        # Compute the reward for each action.  When optimistic_alpha > 0,
        # we blend the simulated reward (partner's current policy) with
        # the max reward over all partner responses (optimistic/cooperative
        # assumption).  This helps the first-acting player learn signaling
        # before the partner has learned to decode.
        action_rewards: list[float] = []
        action_texts: list[str] = []
        for action_id in legal_ids:
          child = state.child(action_id)
          max_reward = None

          # Compute optimistic reward *before* simulation mutates child.
          # _play_out_for_reward applies actions in-place, making child
          # terminal, so the is_terminal() check must happen first.
          if optimistic_alpha > 0 and not child.is_terminal():
            max_reward = self._max_reward_over_partners(
                child, current_player
            )

          sim_reward = self._play_out_for_reward(
              state.child(action_id), current_player
          )

          if optimistic_alpha > 0 and max_reward is not None:
            reward = (
                optimistic_alpha * max_reward
                + (1.0 - optimistic_alpha) * sim_reward
            )
            max_reward = None  # Reset for next iteration.
          else:
            reward = sim_reward

          action_rewards.append(reward)
          # The action text is what the LLM should output to pick this
          # action.  We use the action description (e.g. "Action 0").
          idx = legal_ids.index(action_id)
          action_texts.append(action_descs[idx])

        context_str = ', '.join(context_parts)
        groups.append({
            'player_id': current_player,
            'prompt': prompt,
            'actions': legal_ids,
            'action_texts': action_texts,
            'rewards': action_rewards,
            'context': context_str,
        })

        # Also recurse to find groups for downstream players.
        # For each action, walk the subtree with this player as
        # *not* the target — other players' decision points become
        # the target.
        for action_id in legal_ids:
          child = state.child(action_id)
          action_str = state.action_to_string(current_player, action_id)
          _walk(
              child,
              context_parts + [f'p{current_player}:{action_str}'],
              None,  # Reset target so next player gets groups too.
          )

      else:
        # This is another player's decision point and we already have a
        # target player.  Use the LLM to pick a single action.
        state_text = self._renderers[current_player].render_state(
            state, current_player, game
        )
        legal_actions_with_desc = self._renderers[
            current_player
        ].render_legal_actions(state, current_player, game)
        legal_ids = [a for a, _ in legal_actions_with_desc]
        action_descs = [d for _, d in legal_actions_with_desc]

        prompt = self._agents[current_player]._build_prompt(  # pylint: disable=protected-access
            state_text, legal_ids, action_descs
        )

        with torch.no_grad():
          response, _ = self._backend.generate_with_logprobs(
              prompt,
              temperature=self._current_temperature,
              max_tokens=self._config.max_completion_length,
          )

        action_id = self._renderers[current_player].parse_action(
            response, legal_actions_with_desc
        )
        if action_id is None:
          action_id = int(np.random.choice(legal_ids))

        child = state.child(action_id)
        action_str = state.action_to_string(current_player, action_id)
        _walk(
            child,
            context_parts + [f'p{current_player}:{action_str}'],
            target_player,
        )

    # Start the walk from the initial state.
    initial_state = game.new_initial_state()
    _walk(initial_state, [], None)

    # Deduplicate groups that have identical (player_id, prompt, rewards).
    # This happens when multiple chance-outcome paths lead to the same
    # observable state and the same reward structure.
    seen: set[str] = set()
    unique_groups = []
    for g in groups:
      key = (
          g['player_id'],
          g['prompt'],
          tuple(g['rewards']),
      )
      key_str = str(key)
      if key_str not in seen:
        seen.add(key_str)
        unique_groups.append(g)

    logging.info(
        'Enumerated %d GRPO groups (%d before dedup).',
        len(unique_groups),
        len(groups),
    )
    return unique_groups

  def _play_out_for_reward(
      self,
      state: pyspiel.State,
      target_player: int,
  ) -> float:
    """Play out a game from ``state`` to completion, returning the reward.

    At each remaining decision point, uses the LLM policy (with
    ``torch.no_grad()``) to select actions.  This is used to compute
    deterministic rewards for exhaustive-group GRPO.

    Args:
      state: The game state to play from (not modified in place — the
        method works on a copy).
      target_player: The player whose reward to return.

    Returns:
      The terminal reward for ``target_player``.
    """
    game = self._env.game

    while not state.is_terminal():
      if state.is_chance_node():
        # Should not happen in exhaustive mode — chance nodes are
        # enumerated by _enumerate_grpo_groups.  If we reach one
        # (e.g. in a game with mid-game chance), sample uniformly.
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes)
        action = int(np.random.choice(actions, p=probs))
        state.apply_action(action)
        continue

      current_player = state.current_player()
      state_text = self._renderers[current_player].render_state(
          state, current_player, game
      )
      legal_actions_with_desc = self._renderers[
          current_player
      ].render_legal_actions(state, current_player, game)
      legal_ids = [a for a, _ in legal_actions_with_desc]
      action_descs = [d for _, d in legal_actions_with_desc]

      prompt = self._agents[current_player]._build_prompt(  # pylint: disable=protected-access
          state_text, legal_ids, action_descs
      )

      with torch.no_grad():
        response, _ = self._backend.generate_with_logprobs(
            prompt,
            temperature=self._current_temperature,
            max_tokens=self._config.max_completion_length,
        )

      action_id = self._renderers[current_player].parse_action(
          response, legal_actions_with_desc
      )
      if action_id is None:
        action_id = int(np.random.choice(legal_ids))

      state.apply_action(action_id)

    rewards = state.rewards()
    if rewards is not None:
      return float(rewards[target_player])
    return 0.0

  def _max_reward_over_partners(
      self,
      state: pyspiel.State,
      target_player: int,
  ) -> float:
    """Compute the max reward over all partner actions from ``state``.

    Enumerates all legal actions for the current player at ``state`` and
    recursively finds the maximum achievable reward for ``target_player``.
    This gives an optimistic estimate: "what's the best reward if my
    partner cooperated perfectly?"

    At chance nodes, takes the max over all outcomes (fully optimistic).
    At the target player's own decision nodes (if any remain), also
    takes the max.

    Args:
      state: The game state to evaluate (typically after the target
        player has already acted).
      target_player: The player whose reward to maximize.

    Returns:
      The maximum possible terminal reward for ``target_player``.
    """
    if state.is_terminal():
      rewards = state.rewards()
      return float(rewards[target_player]) if rewards else 0.0

    if state.is_chance_node():
      # Max over all chance outcomes (fully optimistic).
      best = float('-inf')
      for chance_action, _ in state.chance_outcomes():
        child = state.child(chance_action)
        r = self._max_reward_over_partners(child, target_player)
        best = max(best, r)
      return best

    # Player decision node — enumerate all actions and take the max.
    current_player = state.current_player()
    best = float('-inf')
    for action in state.legal_actions(current_player):
      child = state.child(action)
      r = self._max_reward_over_partners(child, target_player)
      best = max(best, r)
    return best

  def _run_exhaustive(self) -> None:
    """Run GRPO training with exhaustive group enumeration.

    For each pass:
      1. Enumerate all GRPO groups by walking the game tree.  Each group
         fixes all variables except one player's action.
      2. For each group, compute log π(a|prompt) for every legal action
         using the backend's differentiable ``compute_action_log_prob``.
      3. Compute group-relative advantages: A_i = r_i - mean(r_group).
      4. Compute the policy gradient loss:
         L = -Σ_i advantage_i * log π(a_i | prompt).
      5. Accumulate gradients across groups and step the optimizer.
      6. Evaluate and log metrics.
    """
    logging.info(
        'Starting exhaustive-group GRPO: %d passes, lr=%.1e',
        self._config.passes,
        self._config.lr,
    )

    # Build optimizer over trainable (LoRA) parameters.
    trainable_params = [
        p for p in self._backend.model.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=self._config.lr
    )

    start_time = time.time()
    total_episodes_so_far = 0

    for pass_idx in range(1, self._config.passes + 1):
      pass_start = time.time()
      logging.info('=== Exhaustive GRPO pass %d/%d ===',
                   pass_idx, self._config.passes)

      # ── Anneal optimistic reward alpha ──
      # Linearly decay from initial alpha toward alpha_min over all passes.
      if self._config.optimistic_reward_alpha > 0:
        progress = (pass_idx - 1) / max(self._config.passes - 1, 1)
        alpha_range = (
            self._config.optimistic_reward_alpha
            - self._config.optimistic_reward_alpha_min
        )
        current_alpha = (
            self._config.optimistic_reward_alpha - alpha_range * progress
        )
        current_alpha = max(
            current_alpha, self._config.optimistic_reward_alpha_min
        )
      else:
        current_alpha = 0.0

      # ── Step 1: Enumerate all groups ──
      self._backend.model.eval()
      groups = self._enumerate_grpo_groups(optimistic_alpha=current_alpha)
      logging.info('  optimistic_alpha=%.3f', current_alpha)

      if not groups:
        logging.warning('No groups enumerated in pass %d.', pass_idx)
        continue

      # ── Pre-compute reference log-probs for KL penalty ──
      # Compute log π_ref(a|s) for all groups at once using the frozen
      # reference weights.  These are constants (no gradient) used in
      # the KL divergence term that prevents the policy from diverging
      # too far from the pre-trained model.
      ref_log_probs_by_group: dict[int, torch.Tensor] = {}
      if self._ref_state_dict is not None and self._config.kl_coeff > 0:
        # Temporarily swap in reference weights.
        current_params = {}
        for name, param in self._backend.model.named_parameters():
          if name in self._ref_state_dict:
            current_params[name] = param.data.clone()
            param.data.copy_(self._ref_state_dict[name])

        self._backend.model.eval()
        with torch.no_grad():
          for gi, group in enumerate(groups):
            ref_lps = []
            for action_text in group['action_texts']:
              ref_lp = self._backend.compute_action_log_prob(
                  group['prompt'], action_text
              )
              ref_lps.append(ref_lp.detach())
            ref_log_probs_by_group[gi] = torch.stack(ref_lps)

        # Restore current weights.
        for name, param in self._backend.model.named_parameters():
          if name in current_params:
            param.data.copy_(current_params[name])

        logging.info(
            '  Pre-computed reference log-probs for %d groups '
            '(kl_coeff=%.3f).',
            len(ref_log_probs_by_group),
            self._config.kl_coeff,
        )

      # ── Pre-compute old log-probs for PPO-style clipping ──
      # Capture the policy's log-probs at the START of this pass
      # (before any gradient updates).  Used to form importance ratios
      # that bound how much the policy can change per pass.
      old_log_probs_by_group: dict[int, torch.Tensor] = {}
      self._backend.model.eval()
      with torch.no_grad():
        for gi, group in enumerate(groups):
          old_lps = []
          for action_text in group['action_texts']:
            old_lp = self._backend.compute_action_log_prob(
                group['prompt'], action_text
            )
            old_lps.append(old_lp.detach())
          old_log_probs_by_group[gi] = torch.stack(old_lps)

      # ── Step 2–4: Compute GRPO loss across all groups ──
      self._backend.model.train()
      optimizer.zero_grad()

      total_loss = 0.0
      total_mean_reward = 0.0
      total_mean_advantage = 0.0
      groups_processed = 0

      for group_idx, group in enumerate(groups):
        player_id = group['player_id']
        prompt = group['prompt']
        action_texts = group['action_texts']
        rewards_list = group['rewards']
        context = group['context']

        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32)
        mean_reward = rewards_tensor.mean().item()

        # Skip groups where all rewards are identical (zero advantage).
        if rewards_tensor.std().item() < 1e-8:
          logging.info(
              '  [Group %d] P%d | %s | all rewards=%.1f, skipping.',
              group_idx, player_id, context, mean_reward,
          )
          continue

        # Compute advantages: A_i = r_i - mean(r).
        advantages = rewards_tensor - rewards_tensor.mean()

        # Compute log π(a_i | prompt) for each action (with gradients).
        log_probs = []
        for action_text in action_texts:
          log_prob = self._backend.compute_action_log_prob(
              prompt, action_text
          )
          log_probs.append(log_prob)

        log_probs_tensor = torch.stack(log_probs)

        # GRPO policy gradient with PPO-style clipping.
        # Normalize advantages by group std for stability.
        advantages_normalized = advantages / (advantages.std() + 1e-8)
        advantages_normalized = advantages_normalized.to(
            log_probs_tensor.device
        )

        # PPO clipped surrogate: use importance ratios r = π_θ / π_old
        # instead of raw log π.  The raw REINFORCE loss -(A * log π) is
        # unbounded as log π → -∞, which no KL penalty can compensate
        # (KL saturates at a finite constant).  Clipping the ratio to
        # [1-ε, 1+ε] bounds the loss and prevents per-pass divergence.
        if group_idx in old_log_probs_by_group:
          old_lps_g = old_log_probs_by_group[group_idx].to(
              log_probs_tensor.device
          )
          ratios = torch.exp(log_probs_tensor - old_lps_g)
          clip_eps = 0.2
          clipped_ratios = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps)
          surr1 = advantages_normalized * ratios
          surr2 = advantages_normalized * clipped_ratios
          group_loss = -torch.min(surr1, surr2).sum()
        else:
          group_loss = -(advantages_normalized * log_probs_tensor).sum()

        # ── KL penalty against reference model ──
        # Long-term regularizer: limits total drift from the pre-trained
        # model across all passes (complementary to per-pass PPO clipping).
        if group_idx in ref_log_probs_by_group:
          ref_lps = ref_log_probs_by_group[group_idx].to(
              log_probs_tensor.device
          )
          probs = torch.softmax(log_probs_tensor, dim=0)
          kl_div = (probs * (log_probs_tensor - ref_lps)).sum()
          group_loss = group_loss + self._config.kl_coeff * kl_div

        # Scale loss for gradient accumulation.
        accum_steps = self._config.gradient_accumulation_steps
        if accum_steps > 1:
          group_loss = group_loss / accum_steps

        group_loss.backward()
        groups_processed += 1

        total_loss += group_loss.item()
        total_mean_reward += mean_reward
        total_mean_advantage += advantages.abs().mean().item()

        logging.info(
            '  [Group %d] P%d | %s | rewards=%s | loss=%.4f',
            group_idx,
            player_id,
            context,
            [f'{r:.0f}' for r in rewards_list],
            group_loss.item(),
        )

        # Step optimizer every gradient_accumulation_steps groups.
        if groups_processed % accum_steps == 0:
          torch.nn.utils.clip_grad_norm_(
              trainable_params, self._config.max_grad_norm
          )
          optimizer.step()
          optimizer.zero_grad()

      # Flush any remaining accumulated gradients.
      if groups_processed % self._config.gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(
            trainable_params, self._config.max_grad_norm
        )
        optimizer.step()
        optimizer.zero_grad()

      # ── Signal entropy bonus ──
      # Maximize entropy of P0's marginal action distribution across
      # game states.  This encourages P0 to use different actions for
      # different cards without pushing probability toward any specific
      # convention.  The entropy gradient is always active regardless
      # of the reward landscape, preventing the policy from collapsing
      # into a non-signaling equilibrium.
      entropy_val = 0.0
      if self._config.signal_entropy_coeff > 0:
        p0_groups = [g for g in groups if g['player_id'] == 0]
        if p0_groups:
          optimizer.zero_grad()
          all_probs = []
          for group in p0_groups:
            prompt = group['prompt']
            action_texts = group['action_texts']
            log_probs = []
            for action_text in action_texts:
              log_prob = self._backend.compute_action_log_prob(
                  prompt, action_text
              )
              log_probs.append(log_prob)
            log_probs_tensor = torch.stack(log_probs)
            probs = torch.softmax(log_probs_tensor, dim=0)
            all_probs.append(probs)

          # Marginal distribution: average over P0 states.
          marginal = torch.stack(all_probs).mean(dim=0)
          marginal = marginal / (marginal.sum() + 1e-10)

          # Entropy: H = -Σ p(a) log p(a).
          entropy = -(marginal * torch.log(marginal + 1e-10)).sum()
          entropy_val = entropy.item()

          # Maximize entropy → minimize -coeff * H.
          entropy_loss = -self._config.signal_entropy_coeff * entropy
          entropy_loss.backward()

          torch.nn.utils.clip_grad_norm_(
              trainable_params, self._config.max_grad_norm
          )
          optimizer.step()
          optimizer.zero_grad()

          logging.info(
              '  Signal entropy: H=%.4f, loss=%.4f, marginal=%s',
              entropy_val,
              entropy_loss.item(),
              [f'{p:.3f}' for p in marginal.detach().cpu().tolist()],
          )

      # Count this pass as a batch of episodes for logging compatibility.
      total_episodes_so_far += len(groups)

      pass_elapsed = time.time() - pass_start
      avg_reward = (
          total_mean_reward / groups_processed if groups_processed else 0.0
      )
      avg_loss = total_loss / groups_processed if groups_processed else 0.0
      logging.info(
          'Pass %d complete: %d groups, avg_loss=%.4f, '
          'avg_reward=%.2f (%.1f sec)',
          pass_idx,
          groups_processed,
          avg_loss,
          avg_reward,
          pass_elapsed,
      )

      # ── Step 5: Log training metrics ──
      if self._log_training_step_fn is not None:
        self._log_training_step_fn(
            pass_idx, avg_reward, avg_loss, start_time
        )

      # ── Step 6: Evaluate ──
      self._backend.model.eval()
      eval_metrics = self._evaluate_fn(self._config.num_eval_episodes)
      logging.info('--- Evaluation after pass %d ---', pass_idx)
      for k, v in sorted(eval_metrics.items()):
        logging.info('  %s: %.4f', k, v)

      if self._log_eval_metrics_fn is not None:
        self._log_eval_metrics_fn(total_episodes_so_far, eval_metrics)

      # ── Step 7: Checkpoint ──
      self._save_checkpoint_fn(total_episodes_so_far)

    # ── Final summary ──
    total_time = time.time() - start_time
    logging.info(
        'Exhaustive GRPO complete: %d passes in %.1f sec.',
        self._config.passes,
        total_time,
    )
    if self._write_summary_fn is not None:
      self._write_summary_fn(total_time)
    self._save_checkpoint_fn(total_episodes_so_far, suffix='final')

  # ════════════════════════════════════════════════════════════════════════
  # Phased (curriculum) exhaustive GRPO
  # ════════════════════════════════════════════════════════════════════════

  def _enumerate_single_player_groups(
      self,
      target_player_id: int,
      other_player_mode: str = 'oracle',
  ) -> list[dict[str, object]]:
    """Enumerate GRPO groups for a single player.

    Unlike ``_enumerate_grpo_groups`` which creates groups for *both*
    players, this method creates groups only for ``target_player_id``.
    The other player's action is determined by ``other_player_mode``:

      - ``'oracle'``: The other player plays the action that maximizes
        ``target_player_id``'s reward (best possible cooperation).
        Used in Phase 1 to give P1 the cleanest learning signal.

      - ``'simulate'``: The other player's action is sampled from the
        LLM using the *other* player's LoRA adapter.  Used in Phase 2
        so P0 learns to signal given P1's actual learned policy.

    Args:
      target_player_id: The player whose groups to enumerate (0 or 1).
      other_player_mode: How to determine the non-target player's action.
        Either ``'oracle'`` or ``'simulate'``.

    Returns:
      A list of group dicts in the same format as
      ``_enumerate_grpo_groups``.
    """
    game = self._env.game
    groups: list[dict[str, object]] = []
    other_player_id = 1 - target_player_id

    def _walk(state, context_parts: list[str]):
      if state.is_terminal():
        return

      if state.is_chance_node():
        for chance_action, _ in state.chance_outcomes():
          child = state.child(chance_action)
          action_str = state.action_to_string(
              pyspiel.PlayerId.CHANCE, chance_action
          )
          _walk(child, context_parts + [f'chance:{action_str}'])
        return

      current_player = state.current_player()

      if current_player == target_player_id:
        # This is the player we're training — expand into a GRPO group.
        state_text = self._renderers[current_player].render_state(
            state, current_player, game
        )
        legal_actions_with_desc = self._renderers[
            current_player
        ].render_legal_actions(state, current_player, game)
        legal_ids = [a for a, _ in legal_actions_with_desc]
        action_descs = [d for _, d in legal_actions_with_desc]

        prompt = self._agents[current_player]._build_prompt(  # pylint: disable=protected-access
            state_text, legal_ids, action_descs
        )

        # Compute rewards for each action.
        action_rewards: list[float] = []
        action_texts: list[str] = []
        for action_id in legal_ids:
          child = state.child(action_id)
          if child.is_terminal():
            rewards = child.rewards()
            reward = float(rewards[target_player_id]) if rewards else 0.0
          elif other_player_mode == 'oracle':
            # Best possible reward assuming perfect partner cooperation.
            reward = self._max_reward_over_partners(child, target_player_id)
          else:
            # Simulate using the other player's learned policy.
            reward = self._simulate_with_adapter(
                child, target_player_id, f'player_{other_player_id}'
            )
          action_rewards.append(reward)
          idx = legal_ids.index(action_id)
          action_texts.append(action_descs[idx])

        context_str = ', '.join(context_parts)
        groups.append({
            'player_id': current_player,
            'prompt': prompt,
            'actions': legal_ids,
            'action_texts': action_texts,
            'rewards': action_rewards,
            'context': context_str,
        })

        # Recurse into each action's subtree.
        for action_id in legal_ids:
          child = state.child(action_id)
          action_str = state.action_to_string(current_player, action_id)
          _walk(
              child,
              context_parts + [f'p{current_player}:{action_str}'],
          )

      else:
        # This is the OTHER player's decision node.
        if other_player_mode == 'oracle':
          # Use only the oracle-optimal action for the other player.
          # This ensures the target player only trains on groups where
          # the other player follows the optimal signaling convention,
          # avoiding conflicting gradients from out-of-policy actions.
          if not hasattr(self, '_oracle_p0_strategy'):
            self._oracle_p0_strategy = self._compute_oracle_p0_strategy()
          oracle_strategy = self._oracle_p0_strategy

          # Determine the other player's card from context.
          # The card is the chance action dealt to this player.
          # In the game tree, chance actions correspond to card deals
          # in player order, so we need to find the deal for this player.
          # We can extract it from the state's history.
          history = state.history()
          # Chance actions are the first num_players entries in history.
          other_card = history[current_player]

          oracle_action = oracle_strategy.get(other_card)
          if oracle_action is not None and oracle_action in state.legal_actions(current_player):
            child = state.child(oracle_action)
            action_str = state.action_to_string(current_player, oracle_action)
            _walk(
                child,
                context_parts + [f'p{current_player}:{action_str}'],
            )
          else:
            # Fallback: if oracle strategy doesn't cover this case,
            # enumerate all actions (shouldn't happen in Tiny Hanabi).
            logging.warning(
                'Oracle strategy missing for P%d card=%d, '
                'falling back to all actions.',
                current_player, other_card,
            )
            for action_id in state.legal_actions(current_player):
              child = state.child(action_id)
              action_str = state.action_to_string(current_player, action_id)
              _walk(
                  child,
                  context_parts + [f'p{current_player}:{action_str}'],
              )
        else:
          # Simulate: use the other player's adapter to pick one action.
          prev_adapter = self._backend.get_active_adapter()
          self._backend.set_active_adapter(f'player_{current_player}')

          state_text = self._renderers[current_player].render_state(
              state, current_player, game
          )
          legal_actions_with_desc = self._renderers[
              current_player
          ].render_legal_actions(state, current_player, game)
          legal_ids = [a for a, _ in legal_actions_with_desc]
          action_descs = [d for _, d in legal_actions_with_desc]
          prompt = self._agents[current_player]._build_prompt(  # pylint: disable=protected-access
              state_text, legal_ids, action_descs
          )

          with torch.no_grad():
            response, _ = self._backend.generate_with_logprobs(
                prompt,
                temperature=0.01,  # Near-greedy for stable simulation.
                max_tokens=self._config.max_completion_length,
            )

          self._backend.set_active_adapter(prev_adapter)

          action_id = self._renderers[current_player].parse_action(
              response, legal_actions_with_desc
          )
          if action_id is None:
            action_id = int(np.random.choice(legal_ids))

          child = state.child(action_id)
          action_str = state.action_to_string(current_player, action_id)
          _walk(
              child,
              context_parts + [f'p{current_player}:{action_str}'],
          )

    initial_state = game.new_initial_state()
    _walk(initial_state, [])

    # Deduplicate groups.
    seen: set[str] = set()
    unique_groups = []
    for g in groups:
      key = (g['player_id'], g['prompt'], tuple(g['rewards']))
      key_str = str(key)
      if key_str not in seen:
        seen.add(key_str)
        unique_groups.append(g)

    logging.info(
        'Enumerated %d %s groups for P%d (%d before dedup).',
        len(unique_groups),
        other_player_mode,
        target_player_id,
        len(groups),
    )
    return unique_groups

  def _simulate_with_adapter(
      self,
      state: pyspiel.State,
      target_player: int,
      other_adapter: str,
  ) -> float:
    """Simulate a game to completion using a specific adapter for the other player.

    Args:
      state: The game state to play from (not modified — works on a copy).
      target_player: The player whose reward to return.
      other_adapter: Name of the adapter to use for the other player.

    Returns:
      The terminal reward for ``target_player``.
    """
    game = self._env.game
    state = state.clone()

    while not state.is_terminal():
      if state.is_chance_node():
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes)
        action = int(np.random.choice(actions, p=probs))
        state.apply_action(action)
        continue

      current_player = state.current_player()
      # Use the appropriate adapter for this player.
      prev_adapter = self._backend.get_active_adapter()
      self._backend.set_active_adapter(
          other_adapter if current_player != target_player
          else f'player_{target_player}'
      )

      state_text = self._renderers[current_player].render_state(
          state, current_player, game
      )
      legal_actions_with_desc = self._renderers[
          current_player
      ].render_legal_actions(state, current_player, game)
      legal_ids = [a for a, _ in legal_actions_with_desc]
      action_descs = [d for _, d in legal_actions_with_desc]
      prompt = self._agents[current_player]._build_prompt(  # pylint: disable=protected-access
          state_text, legal_ids, action_descs
      )

      with torch.no_grad():
        response, _ = self._backend.generate_with_logprobs(
            prompt,
            temperature=0.01,  # Near-greedy.
            max_tokens=self._config.max_completion_length,
        )

      self._backend.set_active_adapter(prev_adapter)

      action_id = self._renderers[current_player].parse_action(
          response, legal_actions_with_desc
      )
      if action_id is None:
        action_id = int(np.random.choice(legal_ids))
      state.apply_action(action_id)

    rewards = state.rewards()
    return float(rewards[target_player]) if rewards else 0.0

  def _compute_oracle_p0_strategy(self) -> dict[int, int]:
    """Compute the optimal P0 signaling strategy for the game.

    Solves for the Stackelberg-optimal P0 pure strategy by brute-forcing
    all possible mappings from P0's card to P0's action, and for each
    one computing P1's best response **given P1's information
    constraint** (P1 observes P0's action and P1's own card, but NOT
    P0's card).

    This is necessary because ``_max_reward_over_partners`` assumes P1
    can see the full state, which makes every P0 action look equally
    good.  The real oracle must account for the fact that P1 must play
    the same action for all P0 cards that map to the same P0 action.

    Returns:
      Dict mapping P0's card (chance-action int) to the optimal P0
      action (int).
    """
    game = self._env.game
    from itertools import product as iter_product  # pylint: disable=g-import-not-at-top

    # ── Step 1: Walk the full game tree to collect all terminal payoffs ──
    # Indexed by (p0_card, p1_card, p0_action, p1_action) -> team_reward.
    payoffs: dict[tuple[int, ...], float] = {}

    def _walk_all(state, deal: list[int], actions: list[int]):
      if state.is_terminal():
        rewards = state.rewards()
        team_r = float(np.mean(rewards)) if rewards else 0.0
        payoffs[tuple(deal + actions)] = team_r
        return
      if state.is_chance_node():
        for action, _ in state.chance_outcomes():
          _walk_all(state.child(action), deal + [action], actions)
        return
      for action in state.legal_actions(state.current_player()):
        _walk_all(state.child(action), deal, actions + [action])

    _walk_all(game.new_initial_state(), [], [])

    # ── Step 2: Extract game dimensions ──
    p0_cards = sorted(set(k[0] for k in payoffs))
    p1_cards = sorted(set(k[1] for k in payoffs))
    p0_actions = sorted(set(k[2] for k in payoffs))
    p1_actions = sorted(set(k[3] for k in payoffs))

    logging.info(
        'Computing optimal P0 oracle: %d P0 cards × %d P0 actions '
        '= %d strategies to check.',
        len(p0_cards), len(p0_actions), len(p0_actions) ** len(p0_cards),
    )

    # ── Step 3: Brute-force all P0 pure strategies ──
    best_strategy: dict[int, int] = {}
    best_expected_reward = float('-inf')
    best_p1_response: dict[tuple[int, int], int] = {}

    for p0_strategy_actions in iter_product(
        p0_actions, repeat=len(p0_cards)
    ):
      p0_map = dict(zip(p0_cards, p0_strategy_actions))

      # For this P0 strategy, compute P1's best response.
      # P1's info state = (p1_card, p0_action).
      # P1 must pick the same action for ALL p0_cards that led to
      # the same p0_action — this is the info-set constraint.
      p1_response: dict[tuple[int, int], int] = {}
      for p1_card in p1_cards:
        for p0_action in set(p0_map.values()):
          p0_cards_for_action = [
              c for c in p0_cards if p0_map[c] == p0_action
          ]

          best_p1_a = p1_actions[0]
          best_p1_r = float('-inf')
          for p1_action in p1_actions:
            expected = float(np.mean([
                payoffs.get(
                    (p0c, p1_card, p0_action, p1_action), 0.0
                )
                for p0c in p0_cards_for_action
            ]))
            if expected > best_p1_r:
              best_p1_r = expected
              best_p1_a = p1_action
          p1_response[(p1_card, p0_action)] = best_p1_a

      # Compute expected reward.
      total = 0.0
      count = 0
      for p0_card in p0_cards:
        for p1_card in p1_cards:
          p0_action = p0_map[p0_card]
          p1_action = p1_response[(p1_card, p0_action)]
          r = payoffs.get(
              (p0_card, p1_card, p0_action, p1_action), 0.0
          )
          total += r
          count += 1

      expected_reward = total / count if count else 0.0

      if expected_reward > best_expected_reward:
        best_expected_reward = expected_reward
        best_strategy = dict(p0_map)
        best_p1_response = dict(p1_response)

    # ── Log the optimal strategy ──
    logging.info(
        'Optimal P0 oracle strategy (expected reward=%.2f):',
        best_expected_reward,
    )
    for card in sorted(best_strategy):
      logging.info('  P0 card=%d → action=%d', card, best_strategy[card])
    logging.info('P1 best response to oracle P0:')
    for (p1_card, p0_action), p1_action in sorted(
        best_p1_response.items()
    ):
      logging.info(
          '  P1 card=%d, saw P0 action=%d → P1 action=%d',
          p1_card, p0_action, p1_action,
      )

    return best_strategy

  def _evaluate_with_oracle(
      self,
      num_episodes: int,
      oracle_player: int,
  ) -> dict[str, float]:
    """Evaluate the trained player against an oracle partner.

    Exhaustively enumerates every card combination and plays each one.
    The oracle player uses the precomputed **optimal signaling strategy**
    (from ``_compute_oracle_p0_strategy``) which correctly accounts for
    P1's information constraint.

    Args:
      num_episodes: Advisory episode count.  The actual number of games
        played is ``num_card_combos * repeats_per_deal``.
      oracle_player: The player that uses oracle-best actions (0 or 1).

    Returns:
      Dict of evaluation metrics including
      ``eval/oracle_mean_reward`` and per-card-combo breakdowns.
    """
    game = self._env.game
    num_players = self._game_config.num_players
    self._backend.model.eval()

    # Compute the oracle strategy (cached after first call).
    if not hasattr(self, '_oracle_p0_strategy'):
      self._oracle_p0_strategy = self._compute_oracle_p0_strategy()
    oracle_strategy = self._oracle_p0_strategy

    # ── Enumerate all card deals ──
    card_deals: list[list[int]] = []

    def _enumerate_deals(state, deal_so_far: list[int]):
      if state.is_chance_node():
        for action, _ in state.chance_outcomes():
          _enumerate_deals(state.child(action), deal_so_far + [action])
      else:
        card_deals.append(deal_so_far)

    _enumerate_deals(game.new_initial_state(), [])

    repeats_per_deal = max(1, num_episodes // max(len(card_deals), 1))
    total_games = len(card_deals) * repeats_per_deal

    logging.info(
        '  Oracle eval: %d card combos × %d repeats = %d games '
        '(P%d=oracle, strategy=%s)',
        len(card_deals), repeats_per_deal, total_games, oracle_player,
        {k: f'a{v}' for k, v in sorted(oracle_strategy.items())},
    )

    all_rewards: list[list[float]] = [[] for _ in range(num_players)]
    per_deal_rewards: dict[str, list[float]] = {}
    action_counts: list[dict[int, int]] = [{} for _ in range(num_players)]

    game_num = 0
    for deal in card_deals:
      deal_key = ','.join(str(d) for d in deal)
      per_deal_rewards[deal_key] = []

      for rep in range(repeats_per_deal):
        game_num += 1
        state = game.new_initial_state()
        for chance_action in deal:
          state.apply_action(chance_action)

        ep_actions: list[str] = ['' for _ in range(num_players)]

        while not state.is_terminal():
          current_player = state.current_player()

          if current_player == oracle_player:
            # Use the precomputed optimal signaling strategy.
            p0_card = deal[oracle_player]
            action_id = oracle_strategy.get(
                p0_card, state.legal_actions(current_player)[0]
            )
            action_str = state.action_to_string(current_player, action_id)
            ep_actions[current_player] = action_str
            action_counts[current_player][action_id] = (
                action_counts[current_player].get(action_id, 0) + 1
            )
            state.apply_action(action_id)

          else:
            # Trained player: use its adapter with near-greedy sampling.
            self._backend.set_active_adapter(f'player_{current_player}')
            state_text = self._renderers[current_player].render_state(
                state, current_player, game
            )
            legal_actions_with_desc = self._renderers[
                current_player
            ].render_legal_actions(state, current_player, game)
            legal_ids = [a for a, _ in legal_actions_with_desc]
            action_descs = [d for _, d in legal_actions_with_desc]
            prompt = self._agents[current_player]._build_prompt(  # pylint: disable=protected-access
                state_text, legal_ids, action_descs
            )

            with torch.no_grad():
              response, _ = self._backend.generate_with_logprobs(
                  prompt,
                  temperature=0.01,
                  max_tokens=self._config.max_completion_length,
              )

            action_id = self._renderers[current_player].parse_action(
                response, legal_actions_with_desc
            )
            if action_id is None:
              action_id = int(np.random.choice(legal_ids))

            action_str = state.action_to_string(current_player, action_id)
            ep_actions[current_player] = action_str
            action_counts[current_player][action_id] = (
                action_counts[current_player].get(action_id, 0) + 1
            )
            state.apply_action(action_id)

        # Collect rewards.
        ep_rewards = state.rewards()
        team_reward = 0.0
        if ep_rewards is not None:
          for p in range(num_players):
            all_rewards[p].append(float(ep_rewards[p]))
          team_reward = float(np.mean(
              [ep_rewards[p] for p in range(num_players)]
          ))
        per_deal_rewards[deal_key].append(team_reward)

        actions_str = ' | '.join(
            f'P{p}:[{ep_actions[p]}]' for p in range(num_players)
        )
        logging.info(
            '  [oracle eval %d/%d] cards=%s | reward=%.1f | %s '
            '(P%d=oracle)',
            game_num, total_games, deal_key, team_reward,
            actions_str, oracle_player,
        )

    # ── Per-deal summary ──
    logging.info('  --- Oracle eval per-deal breakdown ---')
    for deal_key, rewards in sorted(per_deal_rewards.items()):
      mean_r = float(np.mean(rewards))
      logging.info(
          '    cards=%s: mean_reward=%.2f (%s)',
          deal_key, mean_r,
          ', '.join(f'{r:.0f}' for r in rewards),
      )

    # ── Action distributions ──
    for p in range(num_players):
      total = sum(action_counts[p].values())
      if total > 0:
        dist_str = ', '.join(
            f'Action {a}: {c/total*100:.0f}% ({c})'
            for a, c in sorted(action_counts[p].items())
        )
        role = 'oracle' if p == oracle_player else 'trained'
        logging.info(
            '  P%d (%s) action distribution: %s', p, role, dist_str
        )

    # ── Compute metrics ──
    metrics: dict[str, float] = {}
    for p in range(num_players):
      pr = np.array(all_rewards[p])
      metrics[f'eval/mean_reward_p{p}'] = float(np.mean(pr))
    overall = float(
        np.mean([np.mean(all_rewards[p]) for p in range(num_players)])
    )
    metrics['eval/oracle_mean_reward'] = overall

    # Fraction of deals where trained player got optimal reward.
    optimal_count = sum(
        1 for rewards in per_deal_rewards.values()
        if np.mean(rewards) >= 8.0
    )
    metrics['eval/oracle_optimal_deal_frac'] = (
        optimal_count / len(per_deal_rewards) if per_deal_rewards else 0.0
    )

    logging.info(
        '  Oracle eval summary: mean_reward=%.2f, '
        'optimal_deals=%d/%d (%.0f%%)',
        overall, optimal_count, len(per_deal_rewards),
        metrics['eval/oracle_optimal_deal_frac'] * 100,
    )
    return metrics

  def _evaluate_exhaustive_adapters(
      self,
      num_episodes: int,
  ) -> dict[str, float]:
    """Exhaustive deterministic eval with both players using their adapters.

    Enumerates every card combination from the game tree and plays each
    one with both players using their trained LoRA adapters (near-greedy).
    This eliminates variance from random card deals.

    Used for Phase 2+ convergence to ensure P0 learns to signal correctly
    for ALL card combinations, not just lucky random draws.

    Args:
      num_episodes: Advisory count. Actual games = combos × repeats.

    Returns:
      Dict with ``eval/exhaustive_mean_reward`` and
      ``eval/exhaustive_optimal_deal_frac``.
    """
    game = self._env.game
    num_players = self._game_config.num_players
    self._backend.model.eval()

    # ── Enumerate all card deals ──
    card_deals: list[list[int]] = []

    def _enumerate_deals(state, deal_so_far: list[int]):
      if state.is_chance_node():
        for action, _ in state.chance_outcomes():
          _enumerate_deals(state.child(action), deal_so_far + [action])
      else:
        card_deals.append(deal_so_far)

    _enumerate_deals(game.new_initial_state(), [])

    repeats_per_deal = max(1, num_episodes // max(len(card_deals), 1))
    total_games = len(card_deals) * repeats_per_deal

    logging.info(
        '  Exhaustive eval: %d card combos × %d repeats = %d games '
        '(both adapters)',
        len(card_deals), repeats_per_deal, total_games,
    )

    all_rewards: list[list[float]] = [[] for _ in range(num_players)]
    per_deal_rewards: dict[str, list[float]] = {}
    action_counts: list[dict[int, int]] = [{} for _ in range(num_players)]

    game_num = 0
    for deal in card_deals:
      deal_key = ','.join(str(d) for d in deal)
      per_deal_rewards[deal_key] = []

      for rep in range(repeats_per_deal):
        game_num += 1
        state = game.new_initial_state()
        for chance_action in deal:
          state.apply_action(chance_action)

        ep_actions: list[str] = ['' for _ in range(num_players)]

        while not state.is_terminal():
          current_player = state.current_player()
          self._backend.set_active_adapter(f'player_{current_player}')

          state_text = self._renderers[current_player].render_state(
              state, current_player, game
          )
          legal_actions_with_desc = self._renderers[
              current_player
          ].render_legal_actions(state, current_player, game)
          legal_ids = [a for a, _ in legal_actions_with_desc]
          action_descs = [d for _, d in legal_actions_with_desc]
          prompt = self._agents[current_player]._build_prompt(  # pylint: disable=protected-access
              state_text, legal_ids, action_descs
          )

          with torch.no_grad():
            response, _ = self._backend.generate_with_logprobs(
                prompt,
                temperature=0.01,
                max_tokens=self._config.max_completion_length,
            )

          action_id = self._renderers[current_player].parse_action(
              response, legal_actions_with_desc
          )
          if action_id is None:
            action_id = int(np.random.choice(legal_ids))

          action_str = state.action_to_string(current_player, action_id)
          ep_actions[current_player] = action_str
          action_counts[current_player][action_id] = (
              action_counts[current_player].get(action_id, 0) + 1
          )
          state.apply_action(action_id)

        ep_rewards = state.rewards()
        team_reward = 0.0
        if ep_rewards is not None:
          for p in range(num_players):
            all_rewards[p].append(float(ep_rewards[p]))
          team_reward = float(np.mean(
              [ep_rewards[p] for p in range(num_players)]
          ))
        per_deal_rewards[deal_key].append(team_reward)

        actions_str = ' | '.join(
            f'P{p}:[{ep_actions[p]}]' for p in range(num_players)
        )
        logging.info(
            '  [exhaustive eval %d/%d] cards=%s | reward=%.1f | %s',
            game_num, total_games, deal_key, team_reward, actions_str,
        )

    # ── Per-deal summary ──
    logging.info('  --- Exhaustive eval per-deal breakdown ---')
    for deal_key, rewards in sorted(per_deal_rewards.items()):
      mean_r = float(np.mean(rewards))
      logging.info(
          '    cards=%s: mean_reward=%.2f (%s)',
          deal_key, mean_r,
          ', '.join(f'{r:.0f}' for r in rewards),
      )

    for p in range(num_players):
      total = sum(action_counts[p].values())
      if total > 0:
        dist_str = ', '.join(
            f'Action {a}: {c/total*100:.0f}% ({c})'
            for a, c in sorted(action_counts[p].items())
        )
        logging.info('  P%d action distribution: %s', p, dist_str)

    # ── Compute metrics ──
    metrics: dict[str, float] = {}
    for p in range(num_players):
      pr = np.array(all_rewards[p])
      metrics[f'eval/mean_reward_p{p}'] = float(np.mean(pr))
    overall = float(
        np.mean([np.mean(all_rewards[p]) for p in range(num_players)])
    )
    metrics['eval/exhaustive_mean_reward'] = overall

    optimal_count = sum(
        1 for rewards in per_deal_rewards.values()
        if np.mean(rewards) >= 8.0
    )
    metrics['eval/exhaustive_optimal_deal_frac'] = (
        optimal_count / len(per_deal_rewards) if per_deal_rewards else 0.0
    )

    logging.info(
        '  Exhaustive eval summary: mean_reward=%.2f, '
        'optimal_deals=%d/%d (%.0f%%)',
        overall, optimal_count, len(per_deal_rewards),
        metrics['eval/exhaustive_optimal_deal_frac'] * 100,
    )
    return metrics

  def _run_phased_training_pass(
      self,
      phase_name: str,
      target_player: int,
      max_passes: int,
      optimizer: torch.optim.Optimizer,
      total_episodes_so_far: int,
      start_time: float,
      other_player_mode: str = 'oracle',
  ) -> tuple[int, bool]:
    """Run one phase of phased training (train a single player).

    Args:
      phase_name: Label for logging (e.g. 'Phase 1: P1 vs oracle P0').
      target_player: The player to train (0 or 1).
      max_passes: Maximum number of passes for this phase.
      optimizer: The optimizer for this player's adapter.
      total_episodes_so_far: Running episode count for logging.
      start_time: Overall training start time.
      other_player_mode: 'oracle' or 'simulate'.

    Returns:
      Tuple of (updated_total_episodes, converged).
    """
    adapter_name = f'player_{target_player}'
    logging.info(
        '╔══════════════════════════════════════════════════╗'
    )
    logging.info(
        '║ %s', phase_name,
    )
    logging.info(
        '║ Training: P%d adapter=%s, max_passes=%d',
        target_player, adapter_name, max_passes,
    )
    logging.info(
        '║ Other player mode: %s', other_player_mode,
    )
    logging.info(
        '╚══════════════════════════════════════════════════╝'
    )

    # Activate the target player's adapter for training.
    self._backend.set_active_adapter(adapter_name)
    self._backend.unfreeze_adapter(adapter_name)

    best_eval_reward = float('-inf')
    patience_counter = 0
    converged = False

    for pass_idx in range(1, max_passes + 1):
      pass_start = time.time()
      logging.info(
          '=== %s — pass %d/%d ===', phase_name, pass_idx, max_passes
      )

      # Ensure correct adapter is active.
      self._backend.set_active_adapter(adapter_name)

      # ── Enumerate groups for target player only ──
      self._backend.model.eval()
      groups = self._enumerate_single_player_groups(
          target_player, other_player_mode
      )

      if not groups:
        logging.warning('No groups in %s pass %d.', phase_name, pass_idx)
        continue

      # ── Pre-compute reference log-probs for KL penalty ──
      ref_log_probs_by_group: dict[int, torch.Tensor] = {}
      if self._ref_state_dict is not None and self._config.kl_coeff > 0:
        current_params = {}
        for name, param in self._backend.model.named_parameters():
          if name in self._ref_state_dict:
            current_params[name] = param.data.clone()
            param.data.copy_(self._ref_state_dict[name])

        self._backend.model.eval()
        with torch.no_grad():
          for gi, group in enumerate(groups):
            ref_lps = []
            for action_text in group['action_texts']:
              ref_lp = self._backend.compute_action_log_prob(
                  group['prompt'], action_text
              )
              ref_lps.append(ref_lp.detach())
            ref_log_probs_by_group[gi] = torch.stack(ref_lps)

        for name, param in self._backend.model.named_parameters():
          if name in current_params:
            param.data.copy_(current_params[name])

      # ── Pre-compute old log-probs for PPO clipping ──
      old_log_probs_by_group: dict[int, torch.Tensor] = {}
      self._backend.model.eval()
      with torch.no_grad():
        for gi, group in enumerate(groups):
          old_lps = []
          for action_text in group['action_texts']:
            old_lp = self._backend.compute_action_log_prob(
                group['prompt'], action_text
            )
            old_lps.append(old_lp.detach())
          old_log_probs_by_group[gi] = torch.stack(old_lps)

      # ── GRPO loss computation (with inner epochs for small groups) ──
      # When group count is small (e.g. 2 for P0), repeat the gradient
      # computation multiple times per pass to amplify the learning signal.
      # The old_log_probs remain fixed (standard PPO practice).
      inner_epochs = 3 if len(groups) <= 2 else 1
      if inner_epochs > 1:
        logging.info(
            '  Using %d inner epochs (%d groups × %d = %d gradient steps)',
            inner_epochs, len(groups), inner_epochs,
            len(groups) * inner_epochs,
        )

      total_loss = 0.0
      total_mean_reward = 0.0
      groups_processed = 0

      for epoch in range(inner_epochs):
        self._backend.model.train()
        optimizer.zero_grad()

        for group_idx, group in enumerate(groups):
          player_id = group['player_id']
          prompt = group['prompt']
          action_texts = group['action_texts']
          rewards_list = group['rewards']
          context = group['context']

          rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32)
          mean_reward = rewards_tensor.mean().item()

          if rewards_tensor.std().item() < 1e-8:
            if epoch == 0:
              logging.info(
                  '  [Group %d] P%d | %s | all rewards=%.1f, skipping.',
                  group_idx, player_id, context, mean_reward,
              )
            continue

          advantages = rewards_tensor - rewards_tensor.mean()
          log_probs = []
          for action_text in action_texts:
            log_prob = self._backend.compute_action_log_prob(
                prompt, action_text
            )
            log_probs.append(log_prob)
          log_probs_tensor = torch.stack(log_probs)

          # ── Log action probabilities ──
          with torch.no_grad():
            action_probs = torch.softmax(log_probs_tensor, dim=0)
            prob_strs = [
                f'a{i}:{action_probs[i].item()*100:.1f}%'
                for i in range(len(action_probs))
            ]

          advantages_normalized = advantages / (advantages.std() + 1e-8)
          advantages_normalized = advantages_normalized.to(
              log_probs_tensor.device
          )

          # PPO clipped surrogate.
          if group_idx in old_log_probs_by_group:
            old_lps_g = old_log_probs_by_group[group_idx].to(
                log_probs_tensor.device
            )
            ratios = torch.exp(log_probs_tensor - old_lps_g)
            clip_eps = 0.2
            clipped_ratios = torch.clamp(
                ratios, 1.0 - clip_eps, 1.0 + clip_eps
            )
            surr1 = advantages_normalized * ratios
            surr2 = advantages_normalized * clipped_ratios
            group_loss = -torch.min(surr1, surr2).sum()
          else:
            group_loss = -(advantages_normalized * log_probs_tensor).sum()

          # KL penalty.
          if group_idx in ref_log_probs_by_group:
            ref_lps = ref_log_probs_by_group[group_idx].to(
                log_probs_tensor.device
            )
            probs = torch.softmax(log_probs_tensor, dim=0)
            kl_div = (probs * (log_probs_tensor - ref_lps)).sum()
            group_loss = group_loss + self._config.kl_coeff * kl_div

          accum_steps = self._config.gradient_accumulation_steps
          if accum_steps > 1:
            group_loss = group_loss / accum_steps

          group_loss.backward()
          groups_processed += 1
          total_loss += group_loss.item()
          total_mean_reward += mean_reward

          epoch_tag = f' epoch {epoch+1}/{inner_epochs}' if inner_epochs > 1 else ''
          logging.info(
              '  [Group %d%s] P%d | %s | rewards=%s | loss=%.4f | [%s]',
              group_idx, epoch_tag, player_id, context,
              [f'{r:.0f}' for r in rewards_list],
              group_loss.item(),
              ', '.join(prob_strs),
          )

          if groups_processed % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self._backend.model.parameters()
                 if p.requires_grad],
                self._config.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad()

        # Flush remaining gradients at end of each inner epoch.
        if groups_processed % self._config.gradient_accumulation_steps != 0:
          torch.nn.utils.clip_grad_norm_(
              [p for p in self._backend.model.parameters()
               if p.requires_grad],
              self._config.max_grad_norm,
          )
          optimizer.step()
          optimizer.zero_grad()

      total_episodes_so_far += len(groups)
      pass_elapsed = time.time() - pass_start
      avg_reward = (
          total_mean_reward / groups_processed if groups_processed else 0.0
      )
      avg_loss = total_loss / groups_processed if groups_processed else 0.0

      logging.info(
          '%s pass %d complete: %d groups, avg_loss=%.4f, '
          'avg_reward=%.2f (%.1f sec)',
          phase_name, pass_idx, groups_processed, avg_loss,
          avg_reward, pass_elapsed,
      )

      # ── Log training metrics ──
      if self._log_training_step_fn is not None:
        self._log_training_step_fn(
            total_episodes_so_far, avg_reward, avg_loss, start_time
        )

      # ── Evaluate ──
      self._backend.model.eval()

      if other_player_mode == 'oracle':
        # Phase 1: eval P1 against oracle P0 to measure true progress.
        other_player_id = 1 - target_player
        eval_metrics = self._evaluate_with_oracle(
            self._config.num_eval_episodes, oracle_player=other_player_id
        )
        logging.info(
            '--- %s oracle evaluation after pass %d ---',
            phase_name, pass_idx,
        )
      else:
        # Phase 2+: exhaustive deterministic eval with both adapters.
        eval_metrics = self._evaluate_exhaustive_adapters(
            self._config.num_eval_episodes
        )
        logging.info(
            '--- %s exhaustive evaluation after pass %d ---',
            phase_name, pass_idx,
        )
      for k, v in sorted(eval_metrics.items()):
        logging.info('  %s: %.4f', k, v)

      if self._log_eval_metrics_fn is not None:
        self._log_eval_metrics_fn(total_episodes_so_far, eval_metrics)

      # ── Convergence detection ──
      if other_player_mode == 'oracle':
        # Phase 1: require ALL card deals to be optimal before moving on.
        # P1 must get 10 on every combination.
        optimal_frac = eval_metrics.get('eval/oracle_optimal_deal_frac', 0.0)
        eval_reward = eval_metrics.get('eval/oracle_mean_reward', 0.0)

        if optimal_frac >= 1.0:
          logging.info(
              '  ★ %s FULLY CONVERGED after %d passes: '
              'all deals optimal (reward=%.4f).',
              phase_name, pass_idx, eval_reward,
          )
          converged = True
          break

        # Track best for logging, but don't use patience to end the phase.
        if eval_reward > best_eval_reward + self._config.convergence_min_delta:
          best_eval_reward = eval_reward
          patience_counter = 0
          logging.info(
              '  ✓ New best eval reward: %.4f, optimal_deals=%.0f%% '
              '(need 100%%)',
              best_eval_reward, optimal_frac * 100,
          )
        else:
          patience_counter += 1
          if patience_counter >= self._config.convergence_patience:
            logging.info(
                '  ⚠ Reward stalled for %d passes (best=%.4f, '
                'optimal=%.0f%%), but continuing — need 100%% optimal.',
                patience_counter, best_eval_reward, optimal_frac * 100,
            )
            # Don't break — keep training until max_passes or 100%.
          else:
            logging.info(
                '  ✗ No improvement: eval=%.4f, best=%.4f, '
                'optimal=%.0f%%, patience=%d/%d',
                eval_reward, best_eval_reward, optimal_frac * 100,
                patience_counter, self._config.convergence_patience,
            )

      else:
        # Phase 2+: require ALL card deals to be optimal.
        optimal_frac = eval_metrics.get(
            'eval/exhaustive_optimal_deal_frac', 0.0
        )
        eval_reward = eval_metrics.get(
            'eval/exhaustive_mean_reward', 0.0
        )

        if optimal_frac >= 1.0:
          logging.info(
              '  ★ %s FULLY CONVERGED after %d passes: '
              'all deals optimal (reward=%.4f).',
              phase_name, pass_idx, eval_reward,
          )
          converged = True
          break

        if eval_reward > best_eval_reward + self._config.convergence_min_delta:
          best_eval_reward = eval_reward
          patience_counter = 0
          logging.info(
              '  ✓ New best eval reward: %.4f, optimal_deals=%.0f%% '
              '(need 100%%)',
              best_eval_reward, optimal_frac * 100,
          )
        else:
          patience_counter += 1
          if patience_counter >= self._config.convergence_patience:
            logging.info(
                '  ⚠ Reward stalled for %d passes (best=%.4f, '
                'optimal=%.0f%%), but continuing — need 100%% optimal.',
                patience_counter, best_eval_reward, optimal_frac * 100,
            )
          else:
            logging.info(
                '  ✗ No improvement: eval=%.4f, best=%.4f, '
                'optimal=%.0f%%, patience=%d/%d',
                eval_reward, best_eval_reward, optimal_frac * 100,
                patience_counter, self._config.convergence_patience,
            )

      # ── Checkpoint ──
      self._save_checkpoint_fn(total_episodes_so_far)

    if not converged:
      logging.info(
          '  %s reached max passes (%d) without full convergence '
          '(best=%.4f).',
          phase_name, max_passes, best_eval_reward,
      )

    return total_episodes_so_far, converged

  def _run_phased_exhaustive(self) -> None:
    """Run phased (curriculum) exhaustive GRPO training.

    Phase 1: Train P1 only, with P0 playing oracle-best actions.
              P1 learns the optimal response to perfect signals.
    Phase 2: Freeze P1, train P0 only, simulating P1 with its
              learned policy. P0 learns to signal effectively.
    Phase 3: (Optional) Joint fine-tuning with both adapters active.

    Each player has its own LoRA adapter so gradients don't interfere.
    Convergence is detected per-phase using eval reward patience.
    """
    logging.info(
        '╔══════════════════════════════════════════════════════════╗'
    )
    logging.info(
        '║     PHASED EXHAUSTIVE GRPO TRAINING                    ║'
    )
    logging.info(
        '║  Phase 1: Train P1 vs oracle P0  (max %d passes)       ║',
        self._config.phase1_max_passes,
    )
    logging.info(
        '║  Phase 2: Train P0 vs frozen P1  (max %d passes)       ║',
        self._config.phase2_max_passes,
    )
    logging.info(
        '║  Phase 3: Joint fine-tuning      (max %d passes)       ║',
        self._config.phase3_max_passes,
    )
    logging.info(
        '║  Convergence: patience=%d, min_delta=%.2f              ║',
        self._config.convergence_patience,
        self._config.convergence_min_delta,
    )
    logging.info(
        '╚══════════════════════════════════════════════════════════╝'
    )

    start_time = time.time()
    total_episodes_so_far = 0

    # ── Create per-player LoRA adapters ──
    num_players = self._game_config.num_players
    self._backend.create_player_adapters(num_players)

    # ════════════════════════════════════════════════════════════════
    # Phase 1: Train P1 against oracle P0
    # ════════════════════════════════════════════════════════════════
    self._backend.set_active_adapter('player_1')
    self._backend.unfreeze_adapter('player_1')
    p1_trainable = [
        p for p in self._backend.model.parameters() if p.requires_grad
    ]
    p1_optimizer = torch.optim.AdamW(p1_trainable, lr=self._config.lr)

    total_episodes_so_far, p1_converged = self._run_phased_training_pass(
        phase_name='Phase 1: P1 vs oracle P0',
        target_player=1,
        max_passes=self._config.phase1_max_passes,
        optimizer=p1_optimizer,
        total_episodes_so_far=total_episodes_so_far,
        start_time=start_time,
        other_player_mode='oracle',
    )

    phase1_time = time.time() - start_time
    logging.info(
        'Phase 1 complete: %s in %.1f sec.',
        'converged' if p1_converged else 'max passes reached',
        phase1_time,
    )

    # ── Freeze P1's adapter ──
    logging.info('Freezing P1 adapter weights.')
    self._backend.freeze_adapter('player_1')
    self._save_checkpoint_fn(total_episodes_so_far, suffix='phase1_done')

    # ════════════════════════════════════════════════════════════════
    # Phase 2: Train P0 against frozen P1
    # ════════════════════════════════════════════════════════════════
    self._backend.set_active_adapter('player_0')
    self._backend.unfreeze_adapter('player_0')
    p0_trainable = [
        p for p in self._backend.model.parameters() if p.requires_grad
    ]
    p0_optimizer = torch.optim.AdamW(p0_trainable, lr=self._config.lr)

    total_episodes_so_far, p0_converged = self._run_phased_training_pass(
        phase_name='Phase 2: P0 vs frozen P1',
        target_player=0,
        max_passes=self._config.phase2_max_passes,
        optimizer=p0_optimizer,
        total_episodes_so_far=total_episodes_so_far,
        start_time=start_time,
        other_player_mode='simulate',
    )

    phase2_time = time.time() - start_time - phase1_time
    logging.info(
        'Phase 2 complete: %s in %.1f sec.',
        'converged' if p0_converged else 'max passes reached',
        phase2_time,
    )
    self._save_checkpoint_fn(total_episodes_so_far, suffix='phase2_done')

    # ════════════════════════════════════════════════════════════════
    # Phase 3: Joint fine-tuning (optional)
    # ════════════════════════════════════════════════════════════════
    if self._config.phase3_max_passes > 0:
      logging.info(
          '╔══════════════════════════════════════════════════╗'
      )
      logging.info(
          '║ Phase 3: Joint fine-tuning (%d passes)           ║',
          self._config.phase3_max_passes,
      )
      logging.info(
          '╚══════════════════════════════════════════════════╝'
      )

      # Unfreeze both adapters and train them jointly.
      self._backend.unfreeze_adapter('player_0')
      self._backend.unfreeze_adapter('player_1')

      # Train P1 with lower LR, then P0 with lower LR, alternating.
      joint_lr = self._config.lr * 0.1  # Reduced LR for fine-tuning.
      best_eval_reward = float('-inf')
      patience_counter = 0

      for pass_idx in range(1, self._config.phase3_max_passes + 1):
        pass_start = time.time()
        logging.info(
            '=== Phase 3: Joint pass %d/%d (lr=%.1e) ===',
            pass_idx, self._config.phase3_max_passes, joint_lr,
        )

        # Train P1 for one pass.
        self._backend.set_active_adapter('player_1')
        p1_trainable_joint = [
            p for p in self._backend.model.parameters() if p.requires_grad
        ]
        p1_opt_joint = torch.optim.AdamW(p1_trainable_joint, lr=joint_lr)

        groups_p1 = self._enumerate_single_player_groups(1, 'simulate')
        if groups_p1:
          self._train_groups_one_step(
              groups_p1, p1_opt_joint, 'Phase 3 P1'
          )

        # Train P0 for one pass.
        self._backend.set_active_adapter('player_0')
        p0_trainable_joint = [
            p for p in self._backend.model.parameters() if p.requires_grad
        ]
        p0_opt_joint = torch.optim.AdamW(p0_trainable_joint, lr=joint_lr)

        groups_p0 = self._enumerate_single_player_groups(0, 'simulate')
        if groups_p0:
          self._train_groups_one_step(
              groups_p0, p0_opt_joint, 'Phase 3 P0'
          )

        total_episodes_so_far += len(groups_p1) + len(groups_p0)
        pass_elapsed = time.time() - pass_start

        # Evaluate.
        self._backend.model.eval()
        eval_metrics = self._evaluate_fn(self._config.num_eval_episodes)
        logging.info(
            '--- Phase 3 evaluation after pass %d (%.1f sec) ---',
            pass_idx, pass_elapsed,
        )
        for k, v in sorted(eval_metrics.items()):
          logging.info('  %s: %.4f', k, v)

        if self._log_eval_metrics_fn is not None:
          self._log_eval_metrics_fn(total_episodes_so_far, eval_metrics)
        if self._log_training_step_fn is not None:
          self._log_training_step_fn(
              total_episodes_so_far, 0.0, 0.0, start_time
          )

        # Convergence check.
        eval_reward = sum(
            v for k, v in eval_metrics.items()
            if k.startswith('eval/mean_reward')
        ) / max(
            sum(1 for k in eval_metrics if k.startswith('eval/mean_reward')),
            1,
        )
        if eval_reward > best_eval_reward + self._config.convergence_min_delta:
          best_eval_reward = eval_reward
          patience_counter = 0
          logging.info(
              '  ✓ Phase 3 best: %.4f', best_eval_reward,
          )
        else:
          patience_counter += 1
          logging.info(
              '  ✗ Phase 3 patience: %d/%d',
              patience_counter, self._config.convergence_patience,
          )
        if patience_counter >= self._config.convergence_patience:
          logging.info('  ★ Phase 3 converged after %d passes.', pass_idx)
          break

    # ── Final summary ──
    total_time = time.time() - start_time
    logging.info(
        '═══════════════════════════════════════════════════════════'
    )
    logging.info(
        'Phased GRPO complete: %.1f sec total.', total_time,
    )
    logging.info(
        '  Phase 1 (P1 vs oracle): %.1f sec', phase1_time,
    )
    logging.info(
        '  Phase 2 (P0 vs P1):     %.1f sec', phase2_time,
    )
    if self._config.phase3_max_passes > 0:
      logging.info(
          '  Phase 3 (joint):        %.1f sec',
          total_time - phase1_time - phase2_time,
      )
    logging.info(
        '═══════════════════════════════════════════════════════════'
    )

    if self._write_summary_fn is not None:
      self._write_summary_fn(total_time)
    self._save_checkpoint_fn(total_episodes_so_far, suffix='final')

  def _train_groups_one_step(
      self,
      groups: list[dict[str, object]],
      optimizer: torch.optim.Optimizer,
      label: str,
  ) -> float:
    """Train on a list of groups for a single optimizer step.

    Shared helper for Phase 3 joint fine-tuning.

    Args:
      groups: GRPO groups to train on.
      optimizer: The optimizer to use.
      label: Label for logging.

    Returns:
      Average loss across processed groups.
    """
    self._backend.model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    groups_processed = 0

    for group_idx, group in enumerate(groups):
      rewards_list = group['rewards']
      rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32)

      if rewards_tensor.std().item() < 1e-8:
        continue

      advantages = rewards_tensor - rewards_tensor.mean()
      advantages_normalized = advantages / (advantages.std() + 1e-8)

      log_probs = []
      for action_text in group['action_texts']:
        lp = self._backend.compute_action_log_prob(
            group['prompt'], action_text
        )
        log_probs.append(lp)
      log_probs_tensor = torch.stack(log_probs)

      advantages_normalized = advantages_normalized.to(
          log_probs_tensor.device
      )
      group_loss = -(advantages_normalized * log_probs_tensor).sum()
      group_loss.backward()
      groups_processed += 1
      total_loss += group_loss.item()

    if groups_processed > 0:
      torch.nn.utils.clip_grad_norm_(
          [p for p in self._backend.model.parameters() if p.requires_grad],
          self._config.max_grad_norm,
      )
      optimizer.step()
      optimizer.zero_grad()

    avg_loss = total_loss / groups_processed if groups_processed else 0.0
    logging.info(
        '  %s: %d groups, avg_loss=%.4f', label, groups_processed, avg_loss,
    )
    return avg_loss

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
        ser_state = metadata.get('serialized_state', None)

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
              self._simulate_from_state(
                  action_history, action_id, p_id, ser_state
              )
              for _ in range(self._config.reward_num_simulations)
          ]
          mean_r = float(np.mean(sim_rewards))
          std_r = float(np.std(sim_rewards))
          reward = mean_r - self._config.reward_variance_penalty * std_r
        else:
          reward = self._simulate_from_state(
              action_history, action_id, p_id, ser_state
          )
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
