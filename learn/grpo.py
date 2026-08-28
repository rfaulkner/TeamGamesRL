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

"""GRPO (Group Relative Policy Optimization) façade module.

This module is the single public entry point for all GRPO functionality.
External callers should continue to import from here::

  from learn.grpo import GRPOConfig, GRPORunner
  config = GRPOConfig(num_generations=4, passes=10)
  runner = GRPORunner(env, renderers, agents, backend, game_config,
                      evaluate_fn, save_checkpoint_fn, output_dir, config)
  runner.run()

Internally, ``GRPORunner.run()`` dispatches to the appropriate variant:

  - **Sampled** (``grpo_sampled``): TRL-based, for large games.
  - **Exhaustive** (``grpo_exhaustive``): Full game-tree enumeration.
  - **Phased Curriculum** (``grpo_phased``): Three-phase curriculum with
    per-player LoRA adapters.

The algorithm implementations, loss helpers, and game-tree utilities now
live in their own modules under ``learn/``.  See the package structure::

  learn/
  ├── grpo.py              ← you are here (façade)
  ├── grpo_config.py       ← GRPOConfig dataclass
  ├── grpo_loss.py         ← shared loss primitives
  ├── game_tree.py         ← game-tree enumeration + oracle
  ├── grpo_sampled.py      ← sampled (TRL) variant
  ├── grpo_exhaustive.py   ← exhaustive variant
  └── grpo_phased.py       ← phased curriculum variant
"""

from __future__ import annotations

from typing import Any, Callable

from absl import logging

# Re-export GRPOConfig so callers don't need to change imports.
from learn.grpo_config import GRPOConfig


class GRPORunner:
  """Unified GRPO training runner.

  Holds all shared state (environment, renderers, agents, LLM backend,
  configuration, and logging/checkpointing callbacks) and dispatches
  to the appropriate training variant based on ``config`` flags.

  Attributes:
    _env: The OpenSpiel RL environment.
    _renderers: Per-player state renderers.
    _agents: Per-player LLM agent wrappers.
    _backend: The LLM backend (e.g. ``GemmaLLMBackend``).
    _game_config: Game configuration (``GameConfig``).
    _config: The ``GRPOConfig`` instance.
    _evaluate_fn: Evaluation callback.
    _save_checkpoint_fn: Checkpoint-saving callback.
    _output_dir: Base output directory for training artefacts.
    _log_eval_metrics_fn: Optional eval-metric logger.
    _log_training_step_fn: Optional training-step logger.
    _log_episode_fn: Optional episode logger.
    _write_summary_fn: Optional summary writer.
    _update_metrics_fn: Optional in-memory metrics updater.
    _ref_state_dict: Reference model weights for KL penalty.
    _prompt_metadata: Metadata cache for sampled GRPO prompts.
    _current_temperature: Current sampling temperature (may anneal).
    _frozen_lora_state: Snapshot of LoRA weights for partner simulation.
  """

  def __init__(
      self,
      env,
      renderers: list,
      agents: list,
      backend,
      game_config,
      evaluate_fn: Callable[..., dict[str, float]],
      save_checkpoint_fn: Callable[..., None],
      output_dir: str,
      config: GRPOConfig,
      log_eval_metrics_fn: Callable[..., None] | None = None,
      log_training_step_fn: Callable[..., None] | None = None,
      log_episode_fn: Callable[..., None] | None = None,
      write_summary_fn: Callable[..., None] | None = None,
      update_metrics_fn: Callable[..., None] | None = None,
      ref_state_dict: dict[str, Any] | None = None,
  ):
    """Initializes the GRPORunner.

    Args:
      env: An ``rl_environment.Environment`` wrapping the game.
      renderers: List of ``BaseStateRenderer`` instances, one per player.
      agents: List of ``LLMAgent`` instances, one per player.
      backend: An ``LLMBackend`` instance (e.g. ``GemmaLLMBackend``).
      game_config: A ``GameConfig`` describing the game.
      evaluate_fn: ``fn(num_episodes) -> dict[str, float]``.
      save_checkpoint_fn: ``fn(episode_num, suffix=None)``.
      output_dir: Directory for intermediate training outputs.
      config: A ``GRPOConfig`` instance.
      log_eval_metrics_fn: Optional ``fn(episode, metrics_dict)``.
      log_training_step_fn: Optional ``fn(episode, reward, loss, start)``.
      log_episode_fn: Optional ``fn(episode, trajectories, loss, parsed)``.
      write_summary_fn: Optional ``fn(total_time_sec)``.
      update_metrics_fn: Optional ``fn(trajectories, loss)``.
      ref_state_dict: Snapshot of pre-trained model weights for KL
        penalty computation.  If ``None``, KL penalty is disabled.
    """
    self._env = env
    self._renderers = renderers
    self._agents = agents
    self._backend = backend
    self._game_config = game_config
    self._config = config
    self._evaluate_fn = evaluate_fn
    self._save_checkpoint_fn = save_checkpoint_fn
    self._output_dir = output_dir
    self._log_eval_metrics_fn = log_eval_metrics_fn
    self._log_training_step_fn = log_training_step_fn
    self._log_episode_fn = log_episode_fn
    self._write_summary_fn = write_summary_fn
    self._update_metrics_fn = update_metrics_fn
    self._ref_state_dict = ref_state_dict

    # Mutable runtime state.
    self._prompt_metadata: dict[str, dict] = {}
    self._current_temperature: float = config.temperature
    self._frozen_lora_state: dict | None = None

  def run(self) -> None:
    """Run GRPO training, dispatching to the appropriate variant.

    Variant selection:
      - ``exhaustive_groups=True, phased_training=True`` → phased curriculum.
      - ``exhaustive_groups=True`` → exhaustive single-loop.
      - Otherwise → sampled (TRL-based).
    """
    if self._config.exhaustive_groups and self._config.phased_training:
      from learn import grpo_phased  # pylint: disable=g-import-not-at-top

      logging.info('Dispatching to phased curriculum GRPO.')
      grpo_phased.run_phased_exhaustive(self)

    elif self._config.exhaustive_groups:
      from learn import grpo_exhaustive  # pylint: disable=g-import-not-at-top

      logging.info('Dispatching to exhaustive-group GRPO.')
      grpo_exhaustive.run_exhaustive(self)

    else:
      from learn import grpo_sampled  # pylint: disable=g-import-not-at-top

      logging.info('Dispatching to sampled (TRL) GRPO.')
      grpo_sampled.run_sampled(self)
