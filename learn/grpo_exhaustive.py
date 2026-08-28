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

"""Exhaustive-group GRPO runner.

Walks the full game tree to construct GRPO groups where every game
variable except the target player's action is fixed, yielding
zero-variance advantage estimates.  Practical for small games like
Tiny Hanabi (36 terminal states).

All public functions accept a ``runner`` parameter — the ``GRPORunner``
instance that holds shared state.
"""

import time

from absl import logging
from learn import game_tree
from learn import grpo_loss
import torch


def run_exhaustive(runner) -> None:
  """Run GRPO training with exhaustive group enumeration.

  For each pass:
    1. Enumerate all GRPO groups by walking the game tree.
    2. Pre-compute reference and old-policy log-probabilities.
    3. Compute PPO-clipped GRPO loss across all groups.
    4. Apply gradient accumulation and signal entropy bonus.
    5. Evaluate and log metrics.

  Args:
    runner: The ``GRPORunner`` instance.
  """
  logging.info(
      'Starting exhaustive-group GRPO: %d passes, lr=%.1e',
      runner._config.passes, runner._config.lr,
  )

  trainable_params = [
      p for p in runner._backend.model.parameters() if p.requires_grad
  ]
  optimizer = torch.optim.AdamW(trainable_params, lr=runner._config.lr)

  start_time = time.time()
  total_episodes_so_far = 0

  for pass_idx in range(1, runner._config.passes + 1):
    pass_start = time.time()
    logging.info(
        '=== Exhaustive GRPO pass %d/%d ===',
        pass_idx, runner._config.passes,
    )

    # ── Anneal optimistic reward alpha ──
    if runner._config.optimistic_reward_alpha > 0:
      progress = (pass_idx - 1) / max(runner._config.passes - 1, 1)
      alpha_range = (
          runner._config.optimistic_reward_alpha
          - runner._config.optimistic_reward_alpha_min
      )
      current_alpha = max(
          runner._config.optimistic_reward_alpha - alpha_range * progress,
          runner._config.optimistic_reward_alpha_min,
      )
    else:
      current_alpha = 0.0

    # ── Step 1: Enumerate groups ──
    runner._backend.model.eval()
    groups = game_tree.enumerate_grpo_groups(
        runner, optimistic_alpha=current_alpha
    )
    logging.info('  optimistic_alpha=%.3f', current_alpha)

    if not groups:
      logging.warning('No groups enumerated in pass %d.', pass_idx)
      continue

    # ── Pre-compute reference log-probs ──
    ref_lps = {}
    if runner._ref_state_dict is not None and runner._config.kl_coeff > 0:
      ref_lps = grpo_loss.precompute_log_probs(
          groups, runner._backend, state_dict=runner._ref_state_dict
      )
      logging.info(
          '  Pre-computed reference log-probs for %d groups '
          '(kl_coeff=%.3f).',
          len(ref_lps), runner._config.kl_coeff,
      )

    # ── Pre-compute old log-probs ──
    old_lps = grpo_loss.precompute_log_probs(groups, runner._backend)

    # ── GRPO loss across all groups ──
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
            '  [Group %d] P%d | %s | all rewards=%.1f, skipping.',
            gi, group['player_id'], group['context'], mean_reward,
        )
        continue

      # Compute log π(a | s) with gradients.
      log_probs = []
      for action_text in group['action_texts']:
        lp = runner._backend.compute_action_log_prob(
            group['prompt'], action_text
        )
        log_probs.append(lp)
      log_probs_tensor = torch.stack(log_probs)

      group_loss = grpo_loss.compute_group_loss(
          log_probs_tensor,
          adv,
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

      logging.info(
          '  [Group %d] P%d | %s | rewards=%s | loss=%.4f',
          gi, group['player_id'], group['context'],
          [f'{r:.0f}' for r in group['rewards']], group_loss.item(),
      )

      if processed % accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(
            trainable_params, runner._config.max_grad_norm
        )
        optimizer.step()
        optimizer.zero_grad()

    # Flush remaining accumulated gradients.
    if processed % accum_steps != 0:
      torch.nn.utils.clip_grad_norm_(
          trainable_params, runner._config.max_grad_norm
      )
      optimizer.step()
      optimizer.zero_grad()

    # ── Signal entropy bonus ──
    if runner._config.signal_entropy_coeff > 0:
      p0_groups = [g for g in groups if g['player_id'] == 0]
      if p0_groups:
        optimizer.zero_grad()
        entropy, entropy_val = grpo_loss.compute_signal_entropy_loss(
            p0_groups, runner._backend
        )
        entropy_loss = -runner._config.signal_entropy_coeff * entropy
        entropy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            trainable_params, runner._config.max_grad_norm
        )
        optimizer.step()
        optimizer.zero_grad()

        logging.info(
            '  Signal entropy: H=%.4f, loss=%.4f',
            entropy_val, entropy_loss.item(),
        )

    total_episodes_so_far += len(groups)

    pass_elapsed = time.time() - pass_start
    avg_reward = total_mean_reward / processed if processed else 0.0
    avg_loss = total_loss / processed if processed else 0.0
    logging.info(
        'Pass %d complete: %d groups, avg_loss=%.4f, '
        'avg_reward=%.2f (%.1f sec)',
        pass_idx, processed, avg_loss, avg_reward, pass_elapsed,
    )

    # ── Log training metrics ──
    if runner._log_training_step_fn is not None:
      runner._log_training_step_fn(pass_idx, avg_reward, avg_loss, start_time)

    # ── Evaluate ──
    runner._backend.model.eval()
    eval_metrics = runner._evaluate_fn(runner._config.num_eval_episodes)
    logging.info('--- Evaluation after pass %d ---', pass_idx)
    for k, v in sorted(eval_metrics.items()):
      logging.info('  %s: %.4f', k, v)

    if runner._log_eval_metrics_fn is not None:
      runner._log_eval_metrics_fn(total_episodes_so_far, eval_metrics)

    # ── Checkpoint ──
    runner._save_checkpoint_fn(total_episodes_so_far)

  # ── Final summary ──
  total_time = time.time() - start_time
  logging.info(
      'Exhaustive GRPO complete: %d passes in %.1f sec.',
      runner._config.passes, total_time,
  )
  if runner._write_summary_fn is not None:
    runner._write_summary_fn(total_time)
  runner._save_checkpoint_fn(total_episodes_so_far, suffix='final')
