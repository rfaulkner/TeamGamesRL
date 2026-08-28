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

"""Phased curriculum GRPO runner.

Decomposes cooperative learning into a three-phase curriculum:

  Phase 1 — Train P1 against oracle-optimal P0 actions.
  Phase 2 — Freeze P1, train P0 against P1's learned policy.
  Phase 3 — Joint fine-tuning with both LoRA adapters (optional).

Each player gets its own LoRA adapter so gradients don't interfere.
This approach eliminates the chicken-and-egg bootstrapping problem that
causes simultaneous training to converge to non-signaling equilibria.

All public functions accept a ``runner`` parameter — the ``GRPORunner``
instance that holds shared state.
"""

import time

from absl import logging
from learn import game_tree
from learn import grpo_loss
import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════


def evaluate_with_oracle(
    runner,
    num_episodes: int,
    oracle_player: int = 0,
) -> dict[str, float]:
  """Evaluate a trained player against the oracle player.

  The oracle player uses the pre-computed optimal strategy, while the
  trained player uses its current LoRA adapter.

  Args:
    runner: The ``GRPORunner`` instance.
    num_episodes: Number of games to play.
    oracle_player: Which player uses the oracle (default 0).

  Returns:
    Dict of evaluation metrics.
  """
  env = runner._env
  total_rewards = {p: 0.0 for p in range(runner._game_config.num_players)}
  oracle_optimal_deals = 0
  total_deals = 0
  game = env.game

  # Ensure oracle strategy is computed.
  if not hasattr(runner, '_oracle_p0_strategy'):
    runner._oracle_p0_strategy = game_tree.compute_oracle_p0_strategy(runner)

  for _ in range(num_episodes):
    time_step = env.reset()
    while not time_step.last():
      current_player = time_step.current_player()
      state = env._state  # pylint: disable=protected-access

      if current_player == oracle_player:
        history = state.history()
        oracle_card = history[oracle_player]
        action_id = runner._oracle_p0_strategy.get(
            oracle_card,
            state.legal_actions(current_player)[0],
        )
      else:
        state_text = runner._renderers[current_player].render_state(
            state, current_player, game
        )
        legal_actions_with_desc = runner._renderers[
            current_player
        ].render_legal_actions(state, current_player, game)
        legal_ids = [a for a, _ in legal_actions_with_desc]
        action_descs = [d for _, d in legal_actions_with_desc]
        prompt = runner._agents[current_player]._build_prompt(  # pylint: disable=protected-access
            state_text, legal_ids, action_descs
        )

        with torch.no_grad():
          response, _ = runner._backend.generate_with_logprobs(
              prompt, temperature=0.01,
              max_tokens=runner._config.max_completion_length,
          )
        action_id = runner._renderers[current_player].parse_action(
            response, legal_actions_with_desc
        )
        if action_id is None:
          action_id = int(np.random.choice(legal_ids))

      time_step = env.step([action_id])

    if time_step.rewards is not None:
      for p in range(runner._game_config.num_players):
        total_rewards[p] += time_step.rewards[p]

      max_possible = max(
          sum(
              time_step.rewards[p]
              for p in range(runner._game_config.num_players)
          ),
          0.001,
      )
      total_deals += 1
      if sum(time_step.rewards) >= max_possible - 0.001:
        oracle_optimal_deals += 1

  metrics = {}
  for p in range(runner._game_config.num_players):
    metrics[f'eval/mean_reward_p{p}'] = total_rewards[p] / max(
        num_episodes, 1
    )
  metrics['eval/oracle_mean_reward'] = (
      sum(total_rewards.values()) / runner._game_config.num_players
  ) / max(num_episodes, 1)
  metrics['eval/oracle_optimal_deal_frac'] = (
      oracle_optimal_deals / max(total_deals, 1)
  )
  return metrics


def evaluate_exhaustive_adapters(
    runner,
    num_episodes: int,
) -> dict[str, float]:
  """Evaluate both players using their respective LoRA adapters.

  Plays games by switching adapters for each player's decisions,
  producing per-player and aggregate reward metrics.

  Args:
    runner: The ``GRPORunner`` instance.
    num_episodes: Number of games to play.

  Returns:
    Dict of evaluation metrics.
  """
  env = runner._env
  total_rewards = {p: 0.0 for p in range(runner._game_config.num_players)}
  game = env.game

  for _ in range(num_episodes):
    time_step = env.reset()
    while not time_step.last():
      current_player = time_step.current_player()
      state = env._state  # pylint: disable=protected-access

      runner._backend.set_active_adapter(f'player_{current_player}')

      state_text = runner._renderers[current_player].render_state(
          state, current_player, game
      )
      legal_actions_with_desc = runner._renderers[
          current_player
      ].render_legal_actions(state, current_player, game)
      legal_ids = [a for a, _ in legal_actions_with_desc]
      action_descs = [d for _, d in legal_actions_with_desc]
      prompt = runner._agents[current_player]._build_prompt(  # pylint: disable=protected-access
          state_text, legal_ids, action_descs
      )

      with torch.no_grad():
        response, _ = runner._backend.generate_with_logprobs(
            prompt, temperature=0.01,
            max_tokens=runner._config.max_completion_length,
        )
      action_id = runner._renderers[current_player].parse_action(
          response, legal_actions_with_desc
      )
      if action_id is None:
        action_id = int(np.random.choice(legal_ids))

      time_step = env.step([action_id])

    if time_step.rewards is not None:
      for p in range(runner._game_config.num_players):
        total_rewards[p] += time_step.rewards[p]

  metrics = {}
  for p in range(runner._game_config.num_players):
    metrics[f'eval/mean_reward_p{p}'] = total_rewards[p] / max(
        num_episodes, 1
    )
  return metrics


# ═══════════════════════════════════════════════════════════════════════
# Per-phase training pass
# ═══════════════════════════════════════════════════════════════════════


def _run_phased_training_pass(
    runner,
    phase_name: str,
    target_player: int,
    max_passes: int,
    optimizer: torch.optim.Optimizer,
    total_episodes_so_far: int,
    start_time: float,
    other_player_mode: str = 'oracle',
) -> tuple[int, bool]:
  """Run a single-phase training loop for one player.

  This is the core of Phases 1 and 2.  It repeatedly:
    1. Enumerates GRPO groups for the target player.
    2. Pre-computes reference and old log-probs.
    3. Computes PPO-clipped loss + KL penalty.
    4. Evaluates and checks for convergence.

  Args:
    runner: The ``GRPORunner`` instance.
    phase_name: Display name for log messages.
    target_player: Player to train (0 or 1).
    max_passes: Maximum number of passes for this phase.
    optimizer: The optimizer to use.
    total_episodes_so_far: Running episode counter.
    start_time: Training start timestamp.
    other_player_mode: ``'oracle'`` or ``'simulate'``.

  Returns:
    Tuple of ``(updated_total_episodes, converged_bool)``.
  """
  logging.info(
      '╔══════════════════════════════════════════════════╗'
  )
  logging.info('║ %s ║', phase_name.center(48))
  logging.info(
      '╚══════════════════════════════════════════════════╝'
  )

  # Ensure oracle strategy is available.
  oracle_strategy = None
  if other_player_mode == 'oracle':
    if not hasattr(runner, '_oracle_p0_strategy'):
      runner._oracle_p0_strategy = game_tree.compute_oracle_p0_strategy(runner)
    oracle_strategy = runner._oracle_p0_strategy

  best_eval_reward = float('-inf')
  patience_counter = 0

  for pass_idx in range(1, max_passes + 1):
    pass_start = time.time()
    logging.info(
        '=== %s: pass %d/%d ===', phase_name, pass_idx, max_passes,
    )

    # ── Step 1: Enumerate groups ──
    runner._backend.model.eval()
    runner._backend.set_active_adapter(f'player_{target_player}')

    groups = game_tree.enumerate_single_player_groups(
        runner, target_player, other_player_mode, oracle_strategy,
    )

    if not groups:
      logging.warning('No groups enumerated in %s pass %d.', phase_name, pass_idx)
      continue

    # ── Step 2: Pre-compute log-probs ──
    ref_lps = {}
    if runner._ref_state_dict is not None and runner._config.kl_coeff > 0:
      ref_lps = grpo_loss.precompute_log_probs(
          groups, runner._backend, state_dict=runner._ref_state_dict,
      )

    old_lps = grpo_loss.precompute_log_probs(groups, runner._backend)

    # ── Step 3: GRPO loss ──
    runner._backend.model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    total_mean_reward = 0.0
    processed = 0
    accum_steps = runner._config.gradient_accumulation_steps

    for gi, group in enumerate(groups):
      adv, mean_reward, skip = grpo_loss.compute_advantages(group['rewards'])

      if skip:
        logging.info(
            '  [%s group %d] P%d | %s | all rewards=%.1f, skipping.',
            phase_name, gi, group['player_id'],
            group['context'], mean_reward,
        )
        continue

      log_probs = []
      for action_text in group['action_texts']:
        lp = runner._backend.compute_action_log_prob(
            group['prompt'], action_text
        )
        log_probs.append(lp)
      log_probs_tensor = torch.stack(log_probs)

      group_loss = grpo_loss.compute_group_loss(
          log_probs_tensor, adv,
          old_log_probs=old_lps.get(gi),
          ref_log_probs=ref_lps.get(gi),
          kl_coeff=runner._config.kl_coeff,
      )

      if accum_steps > 1:
        group_loss = group_loss / accum_steps

      group_loss.backward()
      processed += 1
      total_loss += group_loss.item()
      total_mean_reward += mean_reward

      # Log action probabilities.
      with torch.no_grad():
        probs = torch.softmax(log_probs_tensor, dim=0)
        prob_strs = [
            f'{group["action_texts"][i]}={probs[i].item():.3f}'
            for i in range(len(probs))
        ]
      logging.info(
          '  [%s group %d] P%d | %s | rewards=%s | '
          'probs=[%s] | loss=%.4f',
          phase_name, gi, group['player_id'], group['context'],
          [f'{r:.0f}' for r in group['rewards']],
          ', '.join(prob_strs), group_loss.item(),
      )

      if processed % accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in runner._backend.model.parameters()
             if p.requires_grad],
            runner._config.max_grad_norm,
        )
        optimizer.step()
        optimizer.zero_grad()

    # Flush remaining gradients.
    if processed > 0 and processed % accum_steps != 0:
      torch.nn.utils.clip_grad_norm_(
          [p for p in runner._backend.model.parameters()
           if p.requires_grad],
          runner._config.max_grad_norm,
      )
      optimizer.step()
      optimizer.zero_grad()

    total_episodes_so_far += len(groups)
    pass_elapsed = time.time() - pass_start

    avg_reward = total_mean_reward / processed if processed else 0.0
    avg_loss = total_loss / processed if processed else 0.0
    logging.info(
        '%s pass %d: %d groups, avg_loss=%.4f, '
        'avg_reward=%.2f (%.1f sec)',
        phase_name, pass_idx, processed, avg_loss, avg_reward, pass_elapsed,
    )

    # ── Step 4: Evaluate ──
    runner._backend.model.eval()
    if other_player_mode == 'oracle':
      eval_metrics = evaluate_with_oracle(
          runner, runner._config.num_eval_episodes,
          oracle_player=1 - target_player,
      )
    else:
      eval_metrics = evaluate_exhaustive_adapters(
          runner, runner._config.num_eval_episodes,
      )

    logging.info('--- %s evaluation after pass %d ---', phase_name, pass_idx)
    for k, v in sorted(eval_metrics.items()):
      logging.info('  %s: %.4f', k, v)

    if runner._log_eval_metrics_fn is not None:
      runner._log_eval_metrics_fn(total_episodes_so_far, eval_metrics)
    if runner._log_training_step_fn is not None:
      runner._log_training_step_fn(
          total_episodes_so_far, avg_reward, avg_loss, start_time
      )

    # ── Convergence check ──
    eval_reward = sum(
        v for k, v in eval_metrics.items()
        if k.startswith('eval/mean_reward')
    ) / max(
        sum(1 for k in eval_metrics if k.startswith('eval/mean_reward')), 1
    )
    if eval_reward > best_eval_reward + runner._config.convergence_min_delta:
      best_eval_reward = eval_reward
      patience_counter = 0
      logging.info('  ✓ New best: %.4f', best_eval_reward)
    else:
      patience_counter += 1
      logging.info(
          '  ✗ Patience: %d/%d',
          patience_counter, runner._config.convergence_patience,
      )
    if patience_counter >= runner._config.convergence_patience:
      logging.info(
          '  ★ %s converged after %d passes.', phase_name, pass_idx,
      )
      return total_episodes_so_far, True

  return total_episodes_so_far, False


# ═══════════════════════════════════════════════════════════════════════
# Main phased training entry point
# ═══════════════════════════════════════════════════════════════════════


def run_phased_exhaustive(runner) -> None:
  """Run the full phased curriculum GRPO training loop.

  Phase 1: Train P1 against oracle P0.
  Phase 2: Freeze P1, train P0 against frozen P1.
  Phase 3: Joint fine-tuning (optional).

  Args:
    runner: The ``GRPORunner`` instance.
  """
  logging.info(
      '╔══════════════════════════════════════════════════════════╗'
  )
  logging.info(
      '║ PHASED EXHAUSTIVE GRPO                                  ║'
  )
  logging.info(
      '║ Phase 1: %3d passes  (P1 vs oracle P0)                  ║',
      runner._config.phase1_max_passes,
  )
  logging.info(
      '║ Phase 2: %3d passes  (P0 vs frozen P1)                  ║',
      runner._config.phase2_max_passes,
  )
  logging.info(
      '║ Phase 3: %3d passes  (joint fine-tuning)                ║',
      runner._config.phase3_max_passes,
  )
  logging.info(
      '╚══════════════════════════════════════════════════════════╝'
  )

  start_time = time.time()
  total_episodes_so_far = 0

  # ── Create per-player LoRA adapters ──
  num_players = runner._game_config.num_players
  runner._backend.create_player_adapters(num_players)

  # ═════════════════════════════════════════════════════════════════════
  # Phase 1: Train P1 against oracle P0
  # ═════════════════════════════════════════════════════════════════════
  runner._backend.set_active_adapter('player_1')
  runner._backend.unfreeze_adapter('player_1')
  p1_trainable = [
      p for p in runner._backend.model.parameters() if p.requires_grad
  ]
  p1_optimizer = torch.optim.AdamW(p1_trainable, lr=runner._config.lr)

  total_episodes_so_far, p1_converged = _run_phased_training_pass(
      runner,
      phase_name='Phase 1: P1 vs oracle P0',
      target_player=1,
      max_passes=runner._config.phase1_max_passes,
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
  runner._backend.freeze_adapter('player_1')
  runner._save_checkpoint_fn(total_episodes_so_far, suffix='phase1_done')

  # ═════════════════════════════════════════════════════════════════════
  # Phase 2: Train P0 against frozen P1
  # ═════════════════════════════════════════════════════════════════════
  runner._backend.set_active_adapter('player_0')
  runner._backend.unfreeze_adapter('player_0')
  p0_trainable = [
      p for p in runner._backend.model.parameters() if p.requires_grad
  ]
  p0_optimizer = torch.optim.AdamW(p0_trainable, lr=runner._config.lr)

  total_episodes_so_far, p0_converged = _run_phased_training_pass(
      runner,
      phase_name='Phase 2: P0 vs frozen P1',
      target_player=0,
      max_passes=runner._config.phase2_max_passes,
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
  runner._save_checkpoint_fn(total_episodes_so_far, suffix='phase2_done')

  # ═════════════════════════════════════════════════════════════════════
  # Phase 3: Joint fine-tuning (optional)
  # ═════════════════════════════════════════════════════════════════════
  if runner._config.phase3_max_passes > 0:
    logging.info(
        '╔══════════════════════════════════════════════════╗'
    )
    logging.info(
        '║ Phase 3: Joint fine-tuning (%d passes)           ║',
        runner._config.phase3_max_passes,
    )
    logging.info(
        '╚══════════════════════════════════════════════════╝'
    )

    runner._backend.unfreeze_adapter('player_0')
    runner._backend.unfreeze_adapter('player_1')

    joint_lr = runner._config.lr * 0.1
    best_eval_reward = float('-inf')
    patience_counter = 0

    for pass_idx in range(1, runner._config.phase3_max_passes + 1):
      pass_start = time.time()
      logging.info(
          '=== Phase 3: Joint pass %d/%d (lr=%.1e) ===',
          pass_idx, runner._config.phase3_max_passes, joint_lr,
      )

      # Train P1 for one pass.
      runner._backend.set_active_adapter('player_1')
      p1_trainable_joint = [
          p for p in runner._backend.model.parameters() if p.requires_grad
      ]
      p1_opt_joint = torch.optim.AdamW(p1_trainable_joint, lr=joint_lr)

      groups_p1 = game_tree.enumerate_single_player_groups(
          runner, 1, 'simulate'
      )
      if groups_p1:
        grpo_loss.train_groups_one_step(
            groups_p1, runner._backend, p1_opt_joint,
            runner._config.max_grad_norm, 'Phase 3 P1',
        )

      # Train P0 for one pass.
      runner._backend.set_active_adapter('player_0')
      p0_trainable_joint = [
          p for p in runner._backend.model.parameters() if p.requires_grad
      ]
      p0_opt_joint = torch.optim.AdamW(p0_trainable_joint, lr=joint_lr)

      groups_p0 = game_tree.enumerate_single_player_groups(
          runner, 0, 'simulate'
      )
      if groups_p0:
        grpo_loss.train_groups_one_step(
            groups_p0, runner._backend, p0_opt_joint,
            runner._config.max_grad_norm, 'Phase 3 P0',
        )

      total_episodes_so_far += len(groups_p1) + len(groups_p0)
      pass_elapsed = time.time() - pass_start

      # Evaluate.
      runner._backend.model.eval()
      eval_metrics = runner._evaluate_fn(runner._config.num_eval_episodes)
      logging.info(
          '--- Phase 3 evaluation after pass %d (%.1f sec) ---',
          pass_idx, pass_elapsed,
      )
      for k, v in sorted(eval_metrics.items()):
        logging.info('  %s: %.4f', k, v)

      if runner._log_eval_metrics_fn is not None:
        runner._log_eval_metrics_fn(total_episodes_so_far, eval_metrics)
      if runner._log_training_step_fn is not None:
        runner._log_training_step_fn(
            total_episodes_so_far, 0.0, 0.0, start_time
        )

      # Convergence check.
      eval_reward = sum(
          v for k, v in eval_metrics.items()
          if k.startswith('eval/mean_reward')
      ) / max(
          sum(1 for k in eval_metrics if k.startswith('eval/mean_reward')), 1
      )
      if eval_reward > best_eval_reward + runner._config.convergence_min_delta:
        best_eval_reward = eval_reward
        patience_counter = 0
        logging.info('  ✓ Phase 3 best: %.4f', best_eval_reward)
      else:
        patience_counter += 1
        logging.info(
            '  ✗ Phase 3 patience: %d/%d',
            patience_counter, runner._config.convergence_patience,
        )
      if patience_counter >= runner._config.convergence_patience:
        logging.info('  ★ Phase 3 converged after %d passes.', pass_idx)
        break

  # ── Final summary ──
  total_time = time.time() - start_time
  logging.info(
      '═══════════════════════════════════════════════════════════'
  )
  logging.info('Phased GRPO complete: %.1f sec total.', total_time)
  logging.info('  Phase 1 (P1 vs oracle): %.1f sec', phase1_time)
  logging.info('  Phase 2 (P0 vs P1):     %.1f sec', phase2_time)
  if runner._config.phase3_max_passes > 0:
    logging.info(
        '  Phase 3 (joint):        %.1f sec',
        total_time - phase1_time - phase2_time,
    )
  logging.info(
      '═══════════════════════════════════════════════════════════'
  )

  if runner._write_summary_fn is not None:
    runner._write_summary_fn(total_time)
  runner._save_checkpoint_fn(total_episodes_so_far, suffix='final')
