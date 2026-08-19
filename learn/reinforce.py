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

"""REINFORCE with baseline and KL penalty for TeamGamesRL.

This module implements the REINFORCE policy-gradient algorithm with:
  - A sliding-window mean baseline to reduce variance.
  - A KL divergence penalty against a frozen reference model to prevent
    mode collapse and language degradation.
  - Gradient accumulation over multiple episodes to further reduce
    REINFORCE variance (~sqrt(N) reduction).

Usage:
  from learn.reinforce import ReinforceConfig, ReinforceUpdater

  config = ReinforceConfig(kl_coeff=0.05, gradient_accumulation_steps=8)
  updater = ReinforceUpdater(backend, optimizer, ref_state_dict, config)
  loss = updater.update(trajectories)
  updater.flush()  # After training loop ends.
"""

import dataclasses

from absl import logging
import numpy as np
import torch

from learn.trajectory import PlayerTrajectory


@dataclasses.dataclass
class ReinforceConfig:
  """Configuration for the REINFORCE algorithm.

  Attributes:
    kl_coeff: KL penalty coefficient against the reference (pre-trained)
        model. Prevents mode collapse and language degradation.
    gradient_accumulation_steps: Number of episodes to accumulate gradients
        over before updating the model. Reduces REINFORCE variance by
        ~sqrt(N).
    baseline_window_size: Number of recent episodes to use for the reward
        baseline (sliding window mean).
    max_grad_norm: Maximum gradient norm for clipping.
  """
  kl_coeff: float = 0.05
  gradient_accumulation_steps: int = 8
  baseline_window_size: int = 50
  max_grad_norm: float = 1.0


class ReinforceUpdater:
  """Stateful REINFORCE loss computation and gradient accumulation.

  This class encapsulates the REINFORCE loss computation with KL penalty,
  reward baseline, and gradient accumulation. It is designed to be called
  once per episode by the training loop.

  The update flow:
    1. Compute advantage = reward - baseline for each player trajectory.
    2. For each (prompt, action_text) pair, compute the differentiable
       log-probability via the backend.
    3. Compute KL penalty against the frozen reference model.
    4. Accumulate scaled gradients.
    5. When enough episodes have been accumulated, clip gradients and
       step the optimizer.

  Attributes:
    _backend: The LLM backend (with compute_action_log_prob, model, device).
    _optimizer: The torch optimizer for trainable parameters.
    _ref_state_dict: Frozen reference state dict for KL penalty.
    _config: The ReinforceConfig.
    _recent_rewards: Sliding window of recent mean rewards for baseline.
    _grad_accum_count: Number of episodes accumulated since last optimizer
        step.
  """

  def __init__(
      self,
      backend,
      optimizer,
      ref_state_dict: dict,
      config: ReinforceConfig,
  ):
    """Initializes the ReinforceUpdater.

    Args:
      backend: Any object with ``compute_action_log_prob(prompt,
          action_text)``, ``model``, and ``device`` attributes.
      optimizer: A torch.optim.Optimizer for the trainable parameters.
      ref_state_dict: A dict of {name: tensor} for the frozen reference
          model parameters (used for KL penalty computation).
      config: A ReinforceConfig instance.
    """
    self._backend = backend
    self._optimizer = optimizer
    self._ref_state_dict = ref_state_dict
    self._config = config
    self._recent_rewards: list[float] = []
    self._grad_accum_count = 0

  def update(self, trajectories: list[PlayerTrajectory]) -> float:
    """Computes REINFORCE loss with KL penalty and accumulates gradients.

    For each player trajectory:
      1. Compute advantage = reward - baseline.
      2. For each step, compute the differentiable log-prob of the action.
      3. Add a KL penalty term against the reference model.
      4. Scale the loss by 1/gradient_accumulation_steps and backprop.

    When ``gradient_accumulation_steps`` episodes have been accumulated,
    clips gradients and steps the optimizer.

    Args:
      trajectories: List of PlayerTrajectory objects from a single episode.

    Returns:
      The scalar REINFORCE loss value (float) for this episode.
    """
    # Update baseline window.
    mean_reward = float(np.mean([t.reward for t in trajectories]))
    self._recent_rewards.append(mean_reward)
    if len(self._recent_rewards) > self._config.baseline_window_size:
      self._recent_rewards = self._recent_rewards[
          -self._config.baseline_window_size :
      ]

    baseline = float(np.mean(self._recent_rewards))

    total_loss_value = 0.0

    for traj in trajectories:
      if not traj.steps:
        continue

      advantage = traj.reward - baseline

      for step in traj.steps:
        # Compute differentiable log-prob.
        log_prob = self._backend.compute_action_log_prob(
            step.prompt, step.action_text
        )

        # REINFORCE loss: -log_prob * advantage.
        reinforce_loss = -log_prob * advantage

        # KL penalty against reference model.
        kl_penalty = torch.tensor(0.0, device=self._backend.device)
        if self._config.kl_coeff > 0:
          for name, param in self._backend.model.named_parameters():
            if name in self._ref_state_dict and param.requires_grad:
              ref_param = self._ref_state_dict[name]
              kl_penalty = kl_penalty + torch.sum(
                  (param - ref_param) ** 2
              )

        loss = reinforce_loss + self._config.kl_coeff * kl_penalty

        # Scale loss for gradient accumulation.
        scaled_loss = loss / self._config.gradient_accumulation_steps
        scaled_loss.backward()

        total_loss_value += float(loss.detach().item())

    self._grad_accum_count += 1

    # Step optimizer if we've accumulated enough gradients.
    if self._grad_accum_count >= self._config.gradient_accumulation_steps:
      torch.nn.utils.clip_grad_norm_(
          self._backend.model.parameters(), self._config.max_grad_norm
      )
      self._optimizer.step()
      self._optimizer.zero_grad()
      self._grad_accum_count = 0
      logging.info(
          'Optimizer step: accumulated %d episodes.',
          self._config.gradient_accumulation_steps,
      )

    return total_loss_value

  def flush(self) -> None:
    """Flushes any remaining accumulated gradients.

    Clips gradients and steps the optimizer if there are accumulated
    gradients from fewer than ``gradient_accumulation_steps`` episodes.
    Should be called at the end of training to ensure no gradients are
    lost.
    """
    if self._grad_accum_count > 0:
      logging.info(
          'Flushing %d accumulated gradient steps.', self._grad_accum_count
      )
      torch.nn.utils.clip_grad_norm_(
          self._backend.model.parameters(), self._config.max_grad_norm
      )
      self._optimizer.step()
      self._optimizer.zero_grad()
      self._grad_accum_count = 0
