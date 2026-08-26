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

"""Model-agnostic RL training orchestrator for multi-agent OpenSpiel games.

This module provides the core training loop that is independent of the
underlying LLM model. It handles:
  - Episode execution (playing games with LLM agents).
  - Evaluation.
  - Checkpointing.
  - Metrics tracking and logging (CSV, JSONL, W&B).

The actual RL algorithm (REINFORCE, GRPO, etc.) is plugged in via the
``learn`` package. The model backend (Gemma, etc.) is plugged in via the
``backend`` package.

Usage:
  from backend.gemma_backend import GemmaLLMBackend
  from learn.reinforce import ReinforceConfig, ReinforceUpdater
  from trainer import RLTrainer

  backend = GemmaLLMBackend(...)
  trainer = RLTrainer(game_name='tiny_hanabi', backend=backend, ...)
  trainer.train_reinforce(reinforce_config)
"""

import json
import os
import time

from absl import logging
import llm_agent
import numpy as np
import torch

from env import game_config as game_config_mod
from env import game_env
from learn.trajectory import PlayerTrajectory
from learn.trajectory import RLTrajectoryStep


class RLTrainer:
  """Model-agnostic RL trainer for multi-agent OpenSpiel games.

  This trainer:
    1. Runs game episodes using LLMAgents backed by any LLMInterface backend.
    2. Collects (prompt, action_text) pairs along with rewards.
    3. Delegates RL loss computation to pluggable algorithm modules.
    4. Handles evaluation, checkpointing, and metrics logging.

  Typical usage:
    ```
    trainer = RLTrainer(game_name='tiny_hanabi', backend=backend, ...)
    trainer.train_reinforce(config)
    # or
    trainer.train_grpo(config)
    ```
  """

  def __init__(
      self,
      game_name: str,
      backend,
      num_episodes: int = 500,
      eval_every: int = 50,
      num_eval_episodes: int = 10,
      lr: float = 1e-4,
      max_grad_norm: float = 1.0,
      temperature: float = 0.8,
      output_dir: str = '/tmp/teamgamesrl',
      log_every: int = 10,
      checkpoint_every: int = 100,
      log_episodes_every: int = 10,
      use_wandb: bool = False,
      wandb_project: str = 'TeamGamesRL',
      wandb_config: dict | None = None,
  ):
    """Initializes the RLTrainer.

    Args:
      game_name: Key into the game configs registry.
      backend: An LLMInterface-compatible backend (e.g., GemmaLLMBackend).
      num_episodes: Total training episodes.
      eval_every: Evaluation frequency (in episodes).
      num_eval_episodes: Number of episodes per evaluation round.
      lr: Learning rate for the optimizer.
      max_grad_norm: Gradient clipping norm.
      temperature: Sampling temperature for LLM action selection.
      output_dir: Directory for logs and checkpoints.
      log_every: Log training metrics every this many episodes.
      checkpoint_every: Save a checkpoint every this many episodes.
      log_episodes_every: Log full episode transcripts every this many episodes.
          Set to 0 to disable.
      use_wandb: Enable Weights & Biases logging.
      wandb_project: Wandb project name.
      wandb_config: Optional dict of config values to log to wandb.

    Raises:
      ValueError: If game_name is not recognized.
    """
    self.game_config = game_config_mod.get_game_config(game_name)
    self.game_name = game_name
    self.num_episodes = num_episodes
    self.eval_every = eval_every
    self.num_eval_episodes = num_eval_episodes
    self.max_grad_norm = max_grad_norm
    self.temperature = temperature
    self.output_dir = output_dir
    self.log_every = log_every
    self.checkpoint_every = checkpoint_every
    self.log_episodes_every = log_episodes_every
    self.use_wandb = use_wandb
    self.wandb_project = wandb_project
    self.wandb_config = wandb_config or {}
    self.backend = backend

    # ── OpenSpiel environment ──
    self.env = game_env.create_env(self.game_config)

    # ── Per-player renderers and agents ──
    self.renderers = []
    self.agents = []
    for pid in range(self.game_config.num_players):
      renderer = game_env.create_renderer(self.game_config)
      self.renderers.append(renderer)
      agent = llm_agent.LLMAgent(
          player_id=pid,
          renderer=renderer,
          llm=backend,
          env=self.env,
          temperature=temperature,
      )
      self.agents.append(agent)

    # ── Optimizer (only trainable params) ──
    trainable_params = [
        p for p in backend.model.parameters() if p.requires_grad
    ]
    self.optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    # ── Metrics ──
    self._episode_rewards: list[float] = []
    self._episode_losses: list[float] = []
    self._player_wins = np.zeros(self.game_config.num_players, dtype=np.int64)
    self._team_wins = 0
    self._total_episodes = 0

    os.makedirs(output_dir, exist_ok=True)

    # ── Results directory for persistent metrics ──
    self.results_dir = os.path.join(output_dir, 'results')
    os.makedirs(self.results_dir, exist_ok=True)

    # Initialize training metrics CSV.
    self._train_csv_path = os.path.join(
        self.results_dir, 'training_metrics.csv'
    )
    with open(self._train_csv_path, 'w') as f:
      f.write('episode,reward,loss,avg_reward,avg_loss,elapsed_sec\n')

    # Initialize eval metrics CSV.
    self._eval_csv_path = os.path.join(self.results_dir, 'eval_metrics.csv')
    # Eval CSV header will be written dynamically on first eval.
    self._eval_csv_header_written = False

    # ── Reference model for KL penalty (frozen copy) ──
    self._ref_state_dict = {
        k: v.detach().clone()
        for k, v in backend.model.named_parameters()
        if v.requires_grad
    }

    logging.info(
        'RLTrainer ready: game=%s, lr=%g, lora_params=%d',
        game_name,
        lr,
        sum(p.numel() for p in trainable_params),
    )

  def run_episode(
      self,
      is_evaluation: bool = False,
  ) -> list[PlayerTrajectory]:
    """Plays one full episode, collecting trajectory data per player.

    Args:
      is_evaluation: If True, use greedy decoding (temperature -> 0).

    Returns:
      List of PlayerTrajectory objects, one per player.
    """
    num_players = self.game_config.num_players
    trajectories = [PlayerTrajectory(player_id=p) for p in range(num_players)]

    time_step = self.env.reset()

    while not time_step.last():
      current_player = time_step.current_player()

      # Render state text.
      state = self.env._state  # pylint: disable=protected-access
      state_text = self.renderers[current_player].render_state(
          state, current_player, self.env.game
      )

      # Get legal actions with descriptions.
      legal_actions_with_desc = self.renderers[
          current_player
      ].render_legal_actions(state, current_player, self.env.game)
      legal_actions = [a for a, _ in legal_actions_with_desc]
      action_descriptions = [d for _, d in legal_actions_with_desc]

      # Build prompt.
      prompt = self.agents[current_player]._build_prompt(
          state_text, legal_actions, action_descriptions
      )

      # Generate action.
      temp = 0.01 if is_evaluation else self.temperature
      response, log_prob = self.backend.generate_with_logprobs(
          prompt, temperature=temp, max_tokens=64
      )

      # Parse action.
      action_id = self.renderers[current_player].parse_action(
          response, legal_actions_with_desc
      )
      if action_id is None:
        action_id = int(np.random.choice(legal_actions))

      action_text = state.action_to_string(current_player, action_id)

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

      time_step = self.env.step([action_id])

    # Assign rewards.
    if time_step.rewards is not None:
      for p in range(num_players):
        trajectories[p].reward = time_step.rewards[p]

    return trajectories

  def evaluate(self, num_episodes: int = 10) -> dict[str, float]:
    """Evaluates the current policy over multiple episodes.

    Args:
      num_episodes: Number of evaluation episodes.

    Returns:
      Dictionary of evaluation metrics.
    """
    self.backend.model.eval()
    num_players = self.game_config.num_players
    all_rewards = [[] for _ in range(num_players)]
    wins = np.zeros(num_players, dtype=np.int64)

    for _ in range(num_episodes):
      trajectories = self.run_episode(is_evaluation=True)
      rewards = [t.reward for t in trajectories]
      for p in range(num_players):
        all_rewards[p].append(rewards[p])
      max_r = max(rewards)
      winners = [p for p in range(num_players) if rewards[p] == max_r]
      if len(winners) == 1:
        wins[winners[0]] += 1

    self.backend.model.train()

    metrics = {}
    for p in range(num_players):
      pr = np.array(all_rewards[p])
      metrics[f'eval/mean_reward_p{p}'] = float(np.mean(pr))
      metrics[f'eval/win_rate_p{p}'] = float(wins[p] / num_episodes)
    return metrics

  def save_checkpoint(self, episode: int = 0, suffix=None) -> str:
    """Saves a LoRA adapter checkpoint.

    Args:
      episode: Current episode number (used in the checkpoint path).
      suffix: Optional suffix override (e.g., 'final').

    Returns:
      Path to the saved checkpoint directory.
    """
    if suffix:
      ckpt_dir = os.path.join(self.output_dir, f'checkpoint_{suffix}')
    else:
      ckpt_dir = os.path.join(self.output_dir, f'checkpoint_ep{episode}')
    self.backend.model.save_pretrained(ckpt_dir)
    self.backend.tokenizer.save_pretrained(ckpt_dir)
    logging.info('Checkpoint saved: %s', ckpt_dir)
    return ckpt_dir

  def _log_episode(
      self,
      episode: int,
      trajectories: list[PlayerTrajectory],
      loss: float,
      is_evaluation: bool = False,
  ) -> None:
    """Logs a full episode transcript to JSONL for visualization.

    Each line in the JSONL file is one episode with per-step details:
    game state, LLM prompt/response, parsed action, reward, etc.

    Args:
      episode: The episode number.
      trajectories: Per-player trajectories from the episode.
      loss: The RL loss for this episode.
      is_evaluation: Whether this was an evaluation episode.
    """
    log_path = os.path.join(self.output_dir, 'episode_log.jsonl')
    record = {
        'episode': episode,
        'game': self.game_name,
        'is_evaluation': is_evaluation,
        'loss': loss,
        'players': [],
    }
    for traj in trajectories:
      player_data = {
          'player_id': traj.player_id,
          'reward': traj.reward,
          'steps': [],
      }
      for step in traj.steps:
        player_data['steps'].append({
            'state_text': step.state_text,
            'llm_response': step.llm_response,
            'game_action': step.game_action_text,
            'action_id': step.action_id,
            'log_prob': step.log_prob,
        })
      record['players'].append(player_data)

    with open(log_path, 'a') as f:
      f.write(json.dumps(record) + '\n')

  def _update_metrics(
      self,
      trajectories: list[PlayerTrajectory],
      loss: float,
  ) -> float:
    """Updates internal metrics accumulators after an episode.

    Args:
      trajectories: Per-player trajectories from the episode.
      loss: The RL loss for this episode.

    Returns:
      The mean reward for the episode.
    """
    ep_rewards = [t.reward for t in trajectories]
    mean_reward = float(np.mean(ep_rewards))
    self._episode_rewards.append(mean_reward)
    self._episode_losses.append(loss)
    self._total_episodes += 1

    # Track per-player wins (for competitive games).
    max_r = max(ep_rewards)
    winners = [
        p
        for p in range(self.game_config.num_players)
        if ep_rewards[p] == max_r
    ]
    if len(winners) == 1:
      self._player_wins[winners[0]] += 1

    # Track team wins for cooperative games (all players share reward).
    if mean_reward >= 8.0:
      self._team_wins += 1

    return mean_reward

  def _log_training_step(
      self,
      ep: int,
      mean_reward: float,
      loss: float,
      start_time: float,
  ) -> None:
    """Logs training metrics to console, CSV, and optionally W&B.

    Args:
      ep: Current episode number.
      mean_reward: Mean reward for this episode.
      loss: Loss for this episode.
      start_time: Training start timestamp.
    """
    elapsed = time.time() - start_time
    avg_r = float(np.mean(self._episode_rewards[-self.log_every:]))
    avg_l = float(np.mean(self._episode_losses[-self.log_every:]))
    logging.info(
        'Ep %d/%d | reward=%.4f (avg=%.4f) | loss=%.4f (avg=%.4f) | '
        '%.1f sec elapsed',
        ep,
        self.num_episodes,
        mean_reward,
        avg_r,
        loss,
        avg_l,
        elapsed,
    )

    # Write training metrics to CSV.
    with open(self._train_csv_path, 'a') as f:
      f.write(
          f'{ep},{mean_reward:.6f},{loss:.6f},'
          f'{avg_r:.6f},{avg_l:.6f},{elapsed:.1f}\n'
      )

    if self.use_wandb:
      import wandb  # pylint: disable=g-import-not-at-top

      wandb.log({
          'episode': ep,
          'reward': mean_reward,
          'avg_reward': avg_r,
          'loss': loss,
          'avg_loss': avg_l,
      })

  def _log_eval_metrics(self, ep: int, eval_metrics: dict[str, float]) -> None:
    """Logs evaluation metrics to console, CSV, and optionally W&B.

    Args:
      ep: Current episode number.
      eval_metrics: Dictionary of evaluation metrics.
    """
    for k, v in sorted(eval_metrics.items()):
      logging.info('  %s: %.4f', k, v)

    # Write eval metrics to CSV.
    with open(self._eval_csv_path, 'a') as f:
      if not self._eval_csv_header_written:
        header = 'episode,' + ','.join(sorted(eval_metrics.keys()))
        f.write(header + '\n')
        self._eval_csv_header_written = True
      vals = ','.join(
          f'{eval_metrics[k]:.6f}' for k in sorted(eval_metrics.keys())
      )
      f.write(f'{ep},{vals}\n')

    if self.use_wandb:
      import wandb  # pylint: disable=g-import-not-at-top

      wandb.log(eval_metrics, step=ep)

  def _write_final_summary(self, total_time: float) -> None:
    """Writes a final summary JSON and logs summary stats.

    Args:
      total_time: Total training time in seconds.
    """
    total_episodes = max(self._total_episodes, 1)
    mean_reward = (
        float(np.mean(self._episode_rewards))
        if self._episode_rewards
        else 0.0
    )
    mean_loss = (
        float(np.mean(self._episode_losses))
        if self._episode_losses
        else 0.0
    )
    last_10_reward = (
        float(np.mean(self._episode_rewards[-10:]))
        if self._episode_rewards
        else 0.0
    )
    last_10_loss = (
        float(np.mean(self._episode_losses[-10:]))
        if self._episode_losses
        else 0.0
    )

    logging.info(
        'Training complete: %d episodes in %.1f seconds.',
        self._total_episodes if self._total_episodes > 0 else self.num_episodes,
        total_time,
    )
    logging.info('Final mean reward: %.4f', mean_reward)
    logging.info('Final mean loss: %.4f', mean_loss)

    if self._total_episodes > 0:
      for p in range(self.game_config.num_players):
        logging.info(
            '  Player %d win rate: %.1f%% (%d/%d)',
            p,
            100.0 * self._player_wins[p] / total_episodes,
            self._player_wins[p],
            self._total_episodes,
        )
      logging.info(
          '  Team win rate (reward >= 8): %.1f%% (%d/%d)',
          100.0 * self._team_wins / total_episodes,
          self._team_wins,
          self._total_episodes,
      )

    summary = {
        'game': self.game_name,
        'num_episodes': (
            self._total_episodes
            if self._total_episodes > 0
            else self.num_episodes
        ),
        'total_time_sec': round(total_time, 1),
        'final_mean_reward': round(mean_reward, 4),
        'final_mean_loss': round(mean_loss, 4),
        'last_10_mean_reward': round(last_10_reward, 4),
        'last_10_mean_loss': round(last_10_loss, 4),
        'player_win_rates': {
            f'player_{p}': round(
                100.0 * self._player_wins[p] / total_episodes, 2
            )
            for p in range(self.game_config.num_players)
        },
        'team_win_rate': (
            round(100.0 * self._team_wins / total_episodes, 2)
            if self._total_episodes > 0
            else 0.0
        ),
    }
    summary_path = os.path.join(self.results_dir, 'summary.json')
    with open(summary_path, 'w') as f:
      json.dump(summary, f, indent=2)
    logging.info('Results written to %s', self.results_dir)

  def train_reinforce(self, reinforce_config) -> None:
    """Runs the main REINFORCE training loop.

    For each episode:
      1. Play an episode and collect trajectories.
      2. Compute REINFORCE loss and update weights via the updater.
      3. Log metrics.
      4. Periodically evaluate and checkpoint.

    Args:
      reinforce_config: A ``ReinforceConfig`` instance from
          ``learn.reinforce``.
    """
    from learn.reinforce import ReinforceUpdater  # pylint: disable=g-import-not-at-top

    logging.info(
        'Starting REINFORCE training: %d episodes on %s',
        self.num_episodes,
        self.game_name,
    )

    updater = ReinforceUpdater(
        backend=self.backend,
        optimizer=self.optimizer,
        ref_state_dict=self._ref_state_dict,
        config=reinforce_config,
    )

    start_time = time.time()

    # Optional W&B init.
    if self.use_wandb:
      import wandb  # pylint: disable=g-import-not-at-top

      wandb.init(
          project=self.wandb_project,
          config=self.wandb_config,
      )

    self.backend.model.train()

    for ep in range(1, self.num_episodes + 1):
      # ── Episode ──
      trajectories = self.run_episode(is_evaluation=False)
      loss = updater.update(trajectories)

      # ── Per-episode progress (lightweight) ──
      mean_r = float(np.mean([t.reward for t in trajectories]))
      ep_elapsed = time.time() - start_time
      actions_str = ' | '.join(
          f'P{t.player_id}:[{",".join(s.game_action_text for s in t.steps)}]'
          for t in trajectories)
      print(f'[ep {ep}/{self.num_episodes}] reward={mean_r:.3f} '
            f'loss={loss:.4f} ({ep_elapsed:.1f}s) {actions_str}', flush=True)

      # ── Episode logging ──
      if self.log_episodes_every > 0 and ep % self.log_episodes_every == 0:
        self._log_episode(ep, trajectories, loss)

      # ── Metrics ──
      mean_reward = self._update_metrics(trajectories, loss)

      # ── Logging ──
      if ep % self.log_every == 0:
        self._log_training_step(ep, mean_reward, loss, start_time)

      # ── Evaluation ──
      if ep % self.eval_every == 0:
        logging.info('--- Evaluation at episode %d ---', ep)
        eval_metrics = self.evaluate(num_episodes=self.num_eval_episodes)
        self._log_eval_metrics(ep, eval_metrics)

      # ── Checkpoint ──
      if ep % self.checkpoint_every == 0:
        self.save_checkpoint(ep)

    # ── Flush remaining accumulated gradients ──
    updater.flush()

    # ── Final summary ──
    total_time = time.time() - start_time
    self._write_final_summary(total_time)
    self.save_checkpoint(self.num_episodes)

    if self.use_wandb:
      import wandb  # pylint: disable=g-import-not-at-top

      wandb.finish()

  def train_grpo(self, grpo_config) -> None:
    """Runs GRPO training using TRL's GRPOTrainer.

    Args:
      grpo_config: A ``GRPOConfig`` instance from ``learn.grpo``.
    """
    from learn.grpo import GRPORunner  # pylint: disable=g-import-not-at-top

    if self.use_wandb:
      import wandb  # pylint: disable=g-import-not-at-top

      wandb.init(
          project=self.wandb_project,
          config=self.wandb_config,
      )

    runner = GRPORunner(
        env=self.env,
        renderers=self.renderers,
        agents=self.agents,
        backend=self.backend,
        game_config=self.game_config,
        evaluate_fn=self.evaluate,
        save_checkpoint_fn=self.save_checkpoint,
        output_dir=self.output_dir,
        config=grpo_config,
        log_eval_metrics_fn=self._log_eval_metrics,
        log_training_step_fn=self._log_training_step,
        log_episode_fn=self._log_episode,
        write_summary_fn=self._write_final_summary,
        update_metrics_fn=self._update_metrics,
    )
    runner.run()

    if self.use_wandb:
      import wandb  # pylint: disable=g-import-not-at-top

      wandb.finish()

