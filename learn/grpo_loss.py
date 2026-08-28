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

"""Shared GRPO loss computation helpers.

This module consolidates the core loss primitives used by all three GRPO
variants (exhaustive, phased curriculum, and sampled).  Each function is
stateless and operates on tensors, making them easy to test and reuse.

Key functions:
  - ``compute_advantages``: group-relative advantage estimation.
  - ``compute_group_loss``: PPO-clipped surrogate + optional KL penalty.
  - ``precompute_log_probs``: batch pre-computation for reference / old
    policy log-probabilities.
  - ``compute_signal_entropy_loss``: cross-state entropy bonus for the
    signaling player.
  - ``train_groups_one_step``: lightweight single-step trainer for
    Phase 3 joint fine-tuning.
"""

from absl import logging
import torch


# ── Advantage computation ──────────────────────────────────────────────


def compute_advantages(
    rewards_list: list[float],
) -> tuple[torch.Tensor, float, bool]:
  """Compute group-relative advantages with normalisation.

  Args:
    rewards_list: Per-action rewards within a single GRPO group.

  Returns:
    A tuple of ``(advantages_normalized, mean_reward, should_skip)``
    where ``should_skip`` is True when all rewards are identical
    (zero-variance group, no useful gradient signal).
  """
  rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32)
  mean_reward = rewards_tensor.mean().item()

  if rewards_tensor.std().item() < 1e-8:
    return rewards_tensor, mean_reward, True

  advantages = rewards_tensor - rewards_tensor.mean()
  advantages_normalized = advantages / (advantages.std() + 1e-8)
  return advantages_normalized, mean_reward, False


# ── Per-group GRPO loss ────────────────────────────────────────────────


def compute_group_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor | None = None,
    ref_log_probs: torch.Tensor | None = None,
    kl_coeff: float = 0.05,
    clip_eps: float = 0.2,
) -> torch.Tensor:
  """Compute the GRPO policy gradient loss for a single group.

  Supports three modes depending on available inputs:

  1. **Plain REINFORCE** (no ``old_log_probs``, no ``ref_log_probs``):
     ``L = -Σ A_i · log π(a_i | s)``

  2. **PPO-clipped** (``old_log_probs`` provided):
     ``L = -Σ min(A · ρ, A · clip(ρ, 1-ε, 1+ε))``
     where ``ρ = π_θ / π_old``.

  3. **PPO-clipped + KL** (both provided):
     Adds ``β · D_KL(π_θ ∥ π_ref)`` to the clipped loss.

  Args:
    log_probs: ``(K,)`` current-policy log-probabilities (with grad).
    advantages: ``(K,)`` normalised group-relative advantages.
    old_log_probs: ``(K,)`` log-probs from the policy at the start of
      the current pass (detached).  ``None`` ⟹ plain REINFORCE.
    ref_log_probs: ``(K,)`` log-probs from the reference (pre-trained)
      model (detached).  ``None`` ⟹ no KL penalty.
    kl_coeff: Weight for the KL penalty term.
    clip_eps: PPO clipping range ``[1-ε, 1+ε]``.

  Returns:
    Scalar loss tensor (with gradient).
  """
  advantages = advantages.to(log_probs.device)

  if old_log_probs is not None:
    old_lps = old_log_probs.to(log_probs.device)
    ratios = torch.exp(log_probs - old_lps)
    clipped = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps)
    loss = -torch.min(advantages * ratios, advantages * clipped).sum()
  else:
    loss = -(advantages * log_probs).sum()

  if ref_log_probs is not None and kl_coeff > 0:
    ref_lps = ref_log_probs.to(log_probs.device)
    probs = torch.softmax(log_probs, dim=0)
    kl_div = (probs * (log_probs - ref_lps)).sum()
    loss = loss + kl_coeff * kl_div

  return loss


# ── Batch log-prob pre-computation ─────────────────────────────────────


def precompute_log_probs(
    groups: list[dict],
    backend,
    state_dict: dict | None = None,
) -> dict[int, torch.Tensor]:
  """Pre-compute detached log-probabilities for all groups.

  When ``state_dict`` is provided the backend's trainable weights are
  temporarily swapped to compute log-probs under a different policy
  (e.g. the reference model) and then restored.

  Args:
    groups: List of GRPO group dicts.
    backend: LLM backend with ``compute_action_log_prob`` and ``model``.
    state_dict: Optional parameter snapshot to swap in before
      computation (used for reference-model KL).

  Returns:
    Dict mapping group index to a ``(K,)`` tensor of detached
    log-probabilities.
  """
  # ── Optionally swap in alternate weights ──
  saved_params: dict | None = None
  if state_dict is not None:
    saved_params = {}
    for name, param in backend.model.named_parameters():
      if name in state_dict:
        saved_params[name] = param.data.clone()
        param.data.copy_(state_dict[name])

  backend.model.eval()
  result: dict[int, torch.Tensor] = {}
  with torch.no_grad():
    for gi, group in enumerate(groups):
      lps = []
      for action_text in group['action_texts']:
        lp = backend.compute_action_log_prob(group['prompt'], action_text)
        lps.append(lp.detach())
      result[gi] = torch.stack(lps)

  # ── Restore original weights ──
  if saved_params is not None:
    for name, param in backend.model.named_parameters():
      if name in saved_params:
        param.data.copy_(saved_params[name])

  return result


# ── Signal entropy bonus ───────────────────────────────────────────────


def compute_signal_entropy_loss(
    p0_groups: list[dict],
    backend,
) -> tuple[torch.Tensor, float]:
  """Compute the cross-state signal entropy for Player 0.

  Maximises the entropy of P0's *marginal* action distribution across
  all game states, encouraging P0 to use different actions for different
  cards without prescribing a specific convention.

  Args:
    p0_groups: GRPO groups for Player 0 only.
    backend: LLM backend.

  Returns:
    Tuple of ``(entropy_loss, entropy_value)`` where the loss should be
    multiplied by ``-signal_entropy_coeff`` and backpropagated.
  """
  all_probs = []
  for group in p0_groups:
    log_probs = []
    for action_text in group['action_texts']:
      lp = backend.compute_action_log_prob(group['prompt'], action_text)
      log_probs.append(lp)
    log_probs_tensor = torch.stack(log_probs)
    probs = torch.softmax(log_probs_tensor, dim=0)
    all_probs.append(probs)

  marginal = torch.stack(all_probs).mean(dim=0)
  marginal = marginal / (marginal.sum() + 1e-10)
  entropy = -(marginal * torch.log(marginal + 1e-10)).sum()
  return entropy, entropy.item()


# ── Lightweight single-step trainer ────────────────────────────────────


def train_groups_one_step(
    groups: list[dict],
    backend,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float,
    label: str,
) -> float:
  """Train on a list of GRPO groups for a single optimiser step.

  Uses plain REINFORCE loss (no PPO clipping, no KL).  Intended for
  Phase 3 joint fine-tuning where small adjustments are sufficient.

  Args:
    groups: GRPO group dicts to train on.
    backend: LLM backend.
    optimizer: Optimiser for the active adapter's parameters.
    max_grad_norm: Gradient clipping threshold.
    label: Label for log messages.

  Returns:
    Average loss across processed groups.
  """
  backend.model.train()
  optimizer.zero_grad()

  total_loss = 0.0
  processed = 0

  for group in groups:
    adv, _, skip = compute_advantages(group['rewards'])
    if skip:
      continue

    log_probs = []
    for action_text in group['action_texts']:
      lp = backend.compute_action_log_prob(group['prompt'], action_text)
      log_probs.append(lp)
    log_probs_tensor = torch.stack(log_probs)

    loss = compute_group_loss(log_probs_tensor, adv)
    loss.backward()
    processed += 1
    total_loss += loss.item()

  if processed > 0:
    torch.nn.utils.clip_grad_norm_(
        [p for p in backend.model.parameters() if p.requires_grad],
        max_grad_norm,
    )
    optimizer.step()
    optimizer.zero_grad()

  avg_loss = total_loss / processed if processed else 0.0
  logging.info('  %s: %d groups, avg_loss=%.4f', label, processed, avg_loss)
  return avg_loss
