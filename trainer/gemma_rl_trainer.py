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

"""Gemma 2B RL trainer for multi-agent OpenSpiel team games.

This is the CLI entry point that wires together:
  - The Gemma LLM backend (backend/gemma_backend.py).
  - The RL training algorithms (learn/reinforce.py, learn/grpo.py).
  - The model-agnostic training orchestrator (trainer.py).
  - The OpenSpiel game environments (env/).

Usage:
  python gemma_rl_trainer.py --game=tiny_hanabi --num_episodes=500
  python gemma_rl_trainer.py --game=hanabi --lora_rank=32 --lr=5e-5
  python gemma_rl_trainer.py --rl_algorithm=grpo --game=tiny_hanabi
"""

import dataclasses

from absl import app
from absl import flags
from absl import logging
from env.game_config import AVAILABLE_GAMES
import numpy as np
import torch

FLAGS = flags.FLAGS

# ============================================================================
# Flags
# ============================================================================

flags.DEFINE_enum(
    'rl_algorithm',
    'grpo',
    ['reinforce', 'grpo'],
    'RL algorithm to use. "reinforce" uses the hand-rolled REINFORCE '
    'with baseline + KL penalty. "grpo" uses TRL\'s GRPOTrainer.',
)
flags.DEFINE_enum(
    'game',
    'tiny_hanabi',
    AVAILABLE_GAMES,
    'Name of the OpenSpiel game to train on.',
)
flags.DEFINE_integer(
    'num_episodes', 500, 'Total number of training episodes to run.'
)
flags.DEFINE_integer(
    'eval_every', 50, 'Run evaluation every this many episodes.'
)
flags.DEFINE_integer(
    'num_eval_episodes', 10, 'Number of episodes per evaluation round.'
)
flags.DEFINE_float(
    'temperature', 0.8, 'Sampling temperature for LLM action selection.'
)
flags.DEFINE_float('lr', 3e-5, 'Learning rate for the LoRA adapter.')
flags.DEFINE_integer('lora_rank', 16, 'LoRA adapter rank.')
flags.DEFINE_integer('lora_alpha', 32, 'LoRA scaling alpha.')
flags.DEFINE_float('lora_dropout', 0.05, 'LoRA dropout probability.')
flags.DEFINE_string(
    'model_name', 'google/gemma-2-2b', 'HuggingFace model ID for Gemma 2B.'
)
flags.DEFINE_bool(
    'use_4bit', True, 'Use 4-bit NF4 quantization for the base model.'
)
flags.DEFINE_string(
    'output_dir',
    '/tmp/teamgamesrl',
    'Directory for checkpoints, logs, and metrics.',
)
flags.DEFINE_integer(
    'log_every', 10, 'Log training metrics every this many episodes.'
)
flags.DEFINE_integer(
    'checkpoint_every', 100, 'Save a LoRA checkpoint every this many episodes.'
)
flags.DEFINE_integer('seed', 42, 'Random seed for reproducibility.')
flags.DEFINE_integer(
    'max_seq_len', 512, 'Maximum sequence length for the model.'
)
flags.DEFINE_float('max_grad_norm', 1.0, 'Maximum gradient norm for clipping.')
flags.DEFINE_bool('use_wandb', False, 'Enable Weights & Biases logging.')
flags.DEFINE_string('wandb_project', 'TeamGamesRL', 'Wandb project name.')
flags.DEFINE_integer(
    'log_episodes_every',
    10,
    'Log full episode transcripts (game state + LLM responses) every this '
    'many episodes. Set to 0 to disable.',
)
flags.DEFINE_float(
    'kl_coeff',
    0.05,
    'KL penalty coefficient against the reference (pre-trained) model. '
    'Prevents mode collapse and language degradation.',
)
flags.DEFINE_integer(
    'gradient_accumulation_steps',
    8,
    'Number of episodes to accumulate gradients over before '
    'updating the model. Reduces REINFORCE variance by ~sqrt(N).',
)
flags.DEFINE_integer(
    'baseline_window_size',
    50,
    'Number of recent episodes to use for the reward baseline '
    '(sliding window mean). Replaces the EMA baseline.',
)

# GRPO-specific flags.
flags.DEFINE_integer(
    'grpo_num_generations',
    8,
    'Number of completions to sample per prompt in GRPO (group size K).',
)
flags.DEFINE_integer(
    'grpo_collect_episodes',
    50,
    'Number of episodes to play for collecting game state prompts '
    'before each GRPO training pass.',
)
flags.DEFINE_integer(
    'grpo_train_epochs', 1, 'Number of training epochs per GRPO pass.'
)
flags.DEFINE_integer(
    'grpo_passes', 25, 'Number of collect-then-train passes for GRPO.'
)
flags.DEFINE_integer(
    'grpo_max_completion_length',
    16,
    'Maximum completion length for GRPO generation.',
)
flags.DEFINE_bool(
    'grpo_exhaustive_groups',
    False,
    'If True, enumerate all possible game states and form GRPO groups '
    'where only the target player\'s action varies. Produces deterministic '
    'advantage estimates. Best for small games (e.g. tiny_hanabi).',
)
flags.DEFINE_float(
    'grpo_optimistic_alpha',
    1.0,
    'Blending weight for optimistic (max-over-partner) rewards. '
    'When 1.0, P0 rewards assume best possible partner cooperation. '
    'Linearly annealed toward alpha_min over training. Only used with '
    '--grpo_exhaustive_groups.',
)
flags.DEFINE_float(
    'grpo_optimistic_alpha_min',
    0.2,
    'Minimum floor for optimistic reward alpha annealing. '
    'Prevents premature collapse into uncoordinated equilibria (e.g. 8.0). '
    'Only used with --grpo_exhaustive_groups.',
)

# ============================================================================
# Entry point
# ============================================================================


def main(argv: list[str]) -> None:
  """Main entry point for Gemma RL training."""
  del argv

  np.random.seed(FLAGS.seed)
  torch.manual_seed(FLAGS.seed)

  logging.info('=== TeamGamesRL — Gemma 2B RL Training ===')
  logging.info('Game: %s', FLAGS.game)
  logging.info('Model: %s (4-bit=%s)', FLAGS.model_name, FLAGS.use_4bit)
  logging.info(
      'LoRA: rank=%d, alpha=%d, dropout=%.2f',
      FLAGS.lora_rank,
      FLAGS.lora_alpha,
      FLAGS.lora_dropout,
  )
  logging.info(
      'Training: episodes=%d, lr=%g, temp=%.2f',
      FLAGS.num_episodes,
      FLAGS.lr,
      FLAGS.temperature,
  )

  # ── Load model ──
  from backend.gemma_backend import GemmaLLMBackend  # pylint: disable=g-import-not-at-top

  backend = GemmaLLMBackend(
      model_name=FLAGS.model_name,
      lora_rank=FLAGS.lora_rank,
      lora_alpha=FLAGS.lora_alpha,
      lora_dropout=FLAGS.lora_dropout,
      use_4bit=FLAGS.use_4bit,
      max_seq_len=FLAGS.max_seq_len,
  )

  # ── Build trainer ──
  from trainer.rl_trainer import RLTrainer  # pylint: disable=g-import-not-at-top

  trainer = RLTrainer(
      game_name=FLAGS.game,
      backend=backend,
      num_episodes=FLAGS.num_episodes,
      eval_every=FLAGS.eval_every,
      num_eval_episodes=FLAGS.num_eval_episodes,
      lr=FLAGS.lr,
      max_grad_norm=FLAGS.max_grad_norm,
      temperature=FLAGS.temperature,
      output_dir=FLAGS.output_dir,
      log_every=FLAGS.log_every,
      checkpoint_every=FLAGS.checkpoint_every,
      log_episodes_every=FLAGS.log_episodes_every,
      use_wandb=FLAGS.use_wandb,
      wandb_project=FLAGS.wandb_project,
      wandb_config={
          'game': FLAGS.game,
          'model': FLAGS.model_name,
          'lora_rank': FLAGS.lora_rank,
          'lr': FLAGS.lr,
          'temperature': FLAGS.temperature,
          'num_episodes': FLAGS.num_episodes,
      },
  )

  # ── Train ──
  if FLAGS.rl_algorithm == 'grpo':
    from learn.grpo import GRPOConfig  # pylint: disable=g-import-not-at-top

    logging.info('Using GRPO (TRL) training.')
    grpo_config = GRPOConfig(
        num_generations=FLAGS.grpo_num_generations,
        collect_episodes=FLAGS.grpo_collect_episodes,
        train_epochs=FLAGS.grpo_train_epochs,
        passes=FLAGS.grpo_passes,
        max_completion_length=FLAGS.grpo_max_completion_length,
        lr=FLAGS.lr,
        kl_coeff=FLAGS.kl_coeff,
        max_grad_norm=FLAGS.max_grad_norm,
        temperature=FLAGS.temperature,
        num_eval_episodes=FLAGS.num_eval_episodes,
        exhaustive_groups=FLAGS.grpo_exhaustive_groups,
        optimistic_reward_alpha=FLAGS.grpo_optimistic_alpha,
        optimistic_reward_alpha_min=FLAGS.grpo_optimistic_alpha_min,
    )
    # ── Tiny Hanabi-specific tuning ──
    # For tiny_hanabi, enable exhaustive-group GRPO by default.  This
    # enumerates all game states and forms groups where only the target
    # player's action varies, producing zero-variance advantage estimates.
    # This replaces the previous approach (temperature annealing, reward
    # variance penalty, partner-weight freezing) which caused policy
    # collapse in longer runs.
    if FLAGS.game == 'tiny_hanabi':
      tiny_hanabi_passes = (
          FLAGS.grpo_passes if FLAGS['grpo_passes'].present else 100
      )
      use_exhaustive = (
          FLAGS.grpo_exhaustive_groups
          if FLAGS['grpo_exhaustive_groups'].present
          else True
      )
      grpo_config = dataclasses.replace(
          grpo_config,
          passes=tiny_hanabi_passes,
          exhaustive_groups=use_exhaustive,
      )
      logging.info(
          'Applied Tiny Hanabi-specific GRPO overrides: passes=%d, '
          'exhaustive_groups=%s, alpha=[%.2f -> %.2f]',
          grpo_config.passes,
          grpo_config.exhaustive_groups,
          grpo_config.optimistic_reward_alpha,
          grpo_config.optimistic_reward_alpha_min,
      )
    trainer.train_grpo(grpo_config)
  else:
    from learn.reinforce import ReinforceConfig  # pylint: disable=g-import-not-at-top

    logging.info('Using REINFORCE training.')
    reinforce_config = ReinforceConfig(
        kl_coeff=FLAGS.kl_coeff,
        gradient_accumulation_steps=FLAGS.gradient_accumulation_steps,
        baseline_window_size=FLAGS.baseline_window_size,
        max_grad_norm=FLAGS.max_grad_norm,
    )
    trainer.train_reinforce(reinforce_config)


if __name__ == '__main__':
  app.run(main)
