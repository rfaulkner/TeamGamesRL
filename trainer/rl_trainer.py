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
      max_history_turns: int | None = 20,
      experiment_config: dict | None = None,
      bot_partner: bool = False,
      bot_type: str = 'belief_lookahead',
      reasoning: bool = False,
      eval_batch_size: int = 4,
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
      max_history_turns: For Hanabi, the maximum number of recent moves
          to show in the prompt. None or 0 shows all moves.
      experiment_config: Optional dict of all experiment hyperparameters
          (model, LoRA, training, GRPO, etc.) to persist in the results
          directory as ``config.json``.  When provided, this config is
          also embedded in the final ``summary.json``.
      bot_partner: Whether to partner with a bot during collection
          and evaluation.
      bot_type: Type of bot partner ('belief_lookahead' or 'safe_play').
      reasoning: Whether to use chain-of-thought reasoning prompts.
      eval_batch_size: Number of evaluation episodes to run concurrently in lockstep.

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
    self.bot_partner = bot_partner
    self.bot_type = bot_type
    self.eval_batch_size = eval_batch_size
    self.max_history_turns = max_history_turns
    self._bot = None

    # ── OpenSpiel environment ──
    self.env = game_env.create_env(self.game_config)

    # ── Per-player renderers and agents ──
    self.renderers = []
    self.agents = []
    for pid in range(self.game_config.num_players):
      renderer = game_env.create_renderer(
          self.game_config,
          max_history_turns=max_history_turns,
      )
      self.renderers.append(renderer)
      agent = llm_agent.LLMAgent(
          player_id=pid,
          renderer=renderer,
          llm=backend,
          env=self.env,
          temperature=temperature,
          reasoning=reasoning,
      )
      self.agents.append(agent)
    self.reasoning = reasoning

    # ── Optimizer (only trainable params) ──
    trainable_params = [
        p for p in backend.model.parameters() if p.requires_grad
    ]
    self.optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=0.01,
    )

    # ── Training state ──
    self._total_episodes = 0
    self._total_steps = 0
    self._episode_rewards: list[float] = []
    self._episode_losses: list[float] = []
    self._player_wins = np.zeros(self.game_config.num_players, dtype=np.int64)
    self._team_wins = 0

    os.makedirs(output_dir, exist_ok=True)

    # ── Results directory for persistent metrics ──
    self.results_dir = os.path.join(output_dir, 'results')
    os.makedirs(self.results_dir, exist_ok=True)

    # ── Persist experiment configuration ──
    self._experiment_config = experiment_config or {}
    if self._experiment_config:
      config_path = os.path.join(self.results_dir, 'config.json')
      with open(config_path, 'w') as f:
        json.dump(self._experiment_config, f, indent=2)
      logging.info('Experiment config written to %s', config_path)
      print(
          f'Experiment config ({len(self._experiment_config)} parameters)'
          f' written to {config_path}',
          flush=True,
      )

    # Initialize training metrics CSV.
    self._train_csv_path = os.path.join(
        self.results_dir, 'training_metrics.csv'
    )
    with open(self._train_csv_path, 'w') as f:
      f.write('episode,reward,loss,avg_reward,avg_loss,elapsed_sec\n')

    # Initialize eval metrics CSV.
    self._eval_csv_path = os.path.join(self.results_dir, 'eval_metrics.csv')
    self._eval_csv_header_written = False

    # ── Reference model for KL penalty (frozen copy) ──
    self._ref_state_dict = {
        k: v.detach().clone()
        for k, v in backend.model.named_parameters()
        if v.requires_grad
    }
    logging.info(
        'Reference LoRA state dict snapshotted (%d tensors).',
        len(self._ref_state_dict),
    )

  def _create_bot(self):
    """Creates the partner bot based on bot_type."""
    bot_type = getattr(self, 'bot_type', 'belief_lookahead')
    if bot_type == 'belief_lookahead':
      try:
        from env.hanabi.belief_expert import SafeBeliefLookaheadPlayer  # pylint: disable=g-import-not-at-top
        return SafeBeliefLookaheadPlayer(self.env.game, n_worlds=1, seed=42)
      except Exception as e:
        logging.warning(
            'Failed to load SafeBeliefLookaheadPlayer: %s, falling back to SafePlayPlayer', e
        )
    try:
      from env.hanabi.heuristic_player import SafePlayPlayer  # pylint: disable=g-import-not-at-top
      return SafePlayPlayer(seed=42)
    except Exception as e:
      logging.warning('Failed to load SafePlayPlayer: %s', e)
      return None

  def run_episode(
      self,
      is_evaluation: bool = False,
      bot_player: int | None = None,
      eval_llm_max_horizon: int | None = None,
  ) -> list[PlayerTrajectory]:
    """Runs a single episode and returns the trajectory.

    Args:
      is_evaluation: If True, uses greedy action selection (temp=0.01).
      bot_player: If set, this player is controlled by the partner bot.
      eval_llm_max_horizon: If set and in evaluation, turns at or beyond this
        turn index are played by the heuristic bot instead of the LLM.

    Returns:
      List of PlayerTrajectory objects, one per player.
    """
    num_players = self.game_config.num_players
    trajectories = [PlayerTrajectory(player_id=p) for p in range(num_players)]

    time_step = self.env.reset()

    while not time_step.last():
      current_player = time_step.current_player()

      # ── Swap LoRA adapter for phased training ──
      if hasattr(self.backend, 'set_active_adapter'):
        adapter_name = f'player_{current_player}'
        try:
          self.backend.set_active_adapter(adapter_name)
        except (ValueError, KeyError):
          pass  # No per-player adapters — use default.

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

      total_turns_so_far = sum(len(t.steps) for t in trajectories)
      use_bot = (bot_player is not None and current_player == bot_player) or (
          is_evaluation
          and eval_llm_max_horizon is not None
          and total_turns_so_far >= eval_llm_max_horizon
      )

      if use_bot:
        if self._bot is None:
          self._bot = self._create_bot()
        if self._bot is not None:
          action_id = self._bot.select_action(state, current_player, self.env.game)
        else:
          action_id = int(np.random.choice(legal_actions))
        response = state.action_to_string(current_player, action_id)
        log_prob = 0.0
        prompt = ''
      else:
        # Build prompt.
        prompt = self.agents[current_player]._build_prompt(
            state_text, legal_actions, action_descriptions
        )

        # Generate action.
        eval_temp = 0.2 if self.reasoning else 0.01
        max_tokens = 200 if self.reasoning else 64
        temp = eval_temp if is_evaluation else self.temperature
        response, log_prob = self.backend.generate_with_logprobs(
            prompt, temperature=temp, max_tokens=max_tokens
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
              action_text=response.strip() if response else '',
              action_id=action_id,
              log_prob=log_prob,
              state_text=state_text,
              llm_response=response if response else '',
              game_action_text=action_text,
          )
      )
      time_step = self.env.step([action_id])

    # Assign rewards.
    if time_step.rewards is not None:
      for p in range(num_players):
        trajectories[p].reward = time_step.rewards[p]

    return trajectories

  def run_episodes_batch(
      self,
      batch_plan: list[tuple[int | None, str]],
      is_evaluation: bool = True,
      eval_llm_max_horizon: int | None = None,
  ) -> list[tuple[list[PlayerTrajectory], object, str]]:
    """Runs a batch of episodes in lockstep to batch LLM generations.

    Args:
      batch_plan: List of (bot_player, mode_label) tuples for this batch.
      is_evaluation: Whether this is evaluation mode.
      eval_llm_max_horizon: If set and in evaluation, turns at or beyond this
        turn index are played by the heuristic bot instead of the LLM.

    Returns:
      List of (trajectories, final_state, mode_label) tuples.
    """
    num_players = self.game_config.num_players
    batch_size = len(batch_plan)

    envs = [game_env.create_env(self.game_config) for _ in range(batch_size)]
    batch_renderers = [
        [
            game_env.create_renderer(
                self.game_config,
                max_history_turns=getattr(self, 'max_history_turns', 20),
            )
            for _ in range(num_players)
        ]
        for _ in range(batch_size)
    ]
    time_steps = [env.reset() for env in envs]
    all_trajectories = [
        [PlayerTrajectory(player_id=p) for p in range(num_players)]
        for _ in range(batch_size)
    ]

    if self._bot is None:
      self._bot = self._create_bot()

    eval_temp = 0.2 if self.reasoning else 0.01
    max_tokens = 200 if self.reasoning else 64
    temp = eval_temp if is_evaluation else self.temperature

    active_indices = list(range(batch_size))

    while active_indices:
      # Advance bot turns until every active game needs the LLM (or terminates).
      llm_pending: list[int] = []
      still_active: list[int] = []

      for idx in active_indices:
        env = envs[idx]
        ts = time_steps[idx]
        bot_player, _ = batch_plan[idx]

        while not ts.last():
          cur_player = ts.current_player()
          total_turns_so_far = sum(len(t.steps) for t in all_trajectories[idx])
          use_bot = (bot_player is not None and cur_player == bot_player) or (
              is_evaluation
              and eval_llm_max_horizon is not None
              and total_turns_so_far >= eval_llm_max_horizon
          )
          if use_bot:
            state = env._state  # pylint: disable=protected-access
            legal_desc = batch_renderers[idx][cur_player].render_legal_actions(
                state, cur_player, env.game
            )
            legal_actions = [a for a, _ in legal_desc]
            if self._bot is not None:
              action_id = self._bot.select_action(state, cur_player, env.game)
            else:
              action_id = int(np.random.choice(legal_actions))

            action_text = state.action_to_string(cur_player, action_id)
            all_trajectories[idx][cur_player].steps.append(
                RLTrajectoryStep(
                    prompt='',
                    action_text=action_text,
                    action_id=action_id,
                    log_prob=0.0,
                    state_text=batch_renderers[idx][cur_player].render_state(
                        state, cur_player, env.game
                    ),
                    llm_response=action_text,
                    game_action_text=action_text,
                )
            )
            ts = env.step([action_id])
            time_steps[idx] = ts
          else:
            break

        if ts.last():
          if ts.rewards is not None:
            for p in range(num_players):
              all_trajectories[idx][p].reward = ts.rewards[p]
        else:
          still_active.append(idx)
          llm_pending.append(idx)

      if not still_active:
        break

      # Build prompts for all games currently waiting on an LLM decision.
      prompts: list[str] = []
      step_metadata: list[tuple[int, int, str, list[tuple[int, str]]]] = []

      for idx in llm_pending:
        env = envs[idx]
        ts = time_steps[idx]
        cur_player = ts.current_player()
        state = env._state  # pylint: disable=protected-access
        state_text = batch_renderers[idx][cur_player].render_state(
            state, cur_player, env.game
        )
        legal_desc = batch_renderers[idx][cur_player].render_legal_actions(
            state, cur_player, env.game
        )
        legal_actions = [a for a, _ in legal_desc]
        action_descriptions = [d for _, d in legal_desc]
        prompt = self.agents[cur_player]._build_prompt(
            state_text, legal_actions, action_descriptions
        )
        prompts.append(prompt)
        step_metadata.append((idx, cur_player, state_text, legal_desc))

      # Run one batched generation across all pending prompts on the GPU.
      if hasattr(self.backend, 'generate_batch'):
        responses = self.backend.generate_batch(
            prompts, temperature=temp, max_tokens=max_tokens
        )
      else:
        responses = [
            self.backend.generate(p, temperature=temp, max_tokens=max_tokens)
            for p in prompts
        ]

      # Step each environment with the generated response.
      next_active: list[int] = []
      for (idx, cur_player, state_text, legal_desc), response in zip(
          step_metadata, responses
      ):
        env = envs[idx]
        state = env._state  # pylint: disable=protected-access
        action_id = batch_renderers[idx][cur_player].parse_action(
            response, legal_desc
        )
        legal_actions = [a for a, _ in legal_desc]
        if action_id is None:
          action_id = int(np.random.choice(legal_actions))

        action_text = state.action_to_string(cur_player, action_id)
        all_trajectories[idx][cur_player].steps.append(
            RLTrajectoryStep(
                prompt=prompts[step_metadata.index((idx, cur_player, state_text, legal_desc))],
                action_text=response.strip() if response else '',
                action_id=action_id,
                log_prob=0.0,
                state_text=state_text,
                llm_response=response if response else '',
                game_action_text=action_text,
            )
        )
        ts = env.step([action_id])
        time_steps[idx] = ts

        if ts.last():
          if ts.rewards is not None:
            for p in range(num_players):
              all_trajectories[idx][p].reward = ts.rewards[p]
        else:
          next_active.append(idx)

      active_indices = next_active

    return [
        (all_trajectories[i], envs[i]._state, batch_plan[i][1])
        for i in range(batch_size)
    ]

  def _process_eval_episode(
      self,
      ep_i: int,
      total_episodes: int,
      trajectories: list[PlayerTrajectory],
      state: object,
      mode_label: str,
      all_rewards: list[list[float]],
      wins: np.ndarray,
      action_counts: list[dict[int, int]],
      mode_rewards: dict[str, list[float]],
      num_players: int,
  ) -> None:
    """Processes, logs, and accumulates metrics for one evaluated episode."""
    rewards = [t.reward for t in trajectories]
    mean_r = float(np.mean(rewards))
    mode_rewards.setdefault(mode_label, []).append(mean_r)

    for p in range(num_players):
      all_rewards[p].append(rewards[p])
      for step in trajectories[p].steps:
        a_id = step.action_id
        action_counts[p][a_id] = action_counts[p].get(a_id, 0) + 1

    max_r = max(rewards)
    winners = [p for p in range(num_players) if rewards[p] == max_r]
    if len(winners) == 1:
      wins[winners[0]] += 1

    actions_summary = []
    for t in trajectories:
      steps_summary = ','.join(
          s.game_action_text or f'a{s.action_id}' for s in t.steps
      )
      actions_summary.append(f'P{t.player_id}:[{steps_summary}]')

    cards_str = ''
    if hasattr(state, 'history') and callable(state.history):
      history = state.history()
      if len(history) >= num_players:
        cards = [history[p] for p in range(num_players)]
        cards_str = ' | cards=' + ','.join(str(c) for c in cards)

    logging.info(
        '  [eval %d/%d] (%s) reward=%.1f%s | %s',
        ep_i + 1,
        total_episodes,
        mode_label,
        mean_r,
        cards_str,
        ' | '.join(actions_summary),
    )

    if self.log_episodes_every > 0:
      self._log_episode(ep_i + 1, trajectories, 0.0, is_evaluation=True)

    eval_log_path = os.path.join(self.results_dir, 'eval_episodes.jsonl')
    eval_record = {
        'eval_episode': ep_i + 1,
        'mode': mode_label,
        'game': self.game_name,
        'mean_reward': mean_r,
        'players': [],
    }
    for traj in trajectories:
      p_data = {
          'player_id': traj.player_id,
          'reward': traj.reward,
          'steps': [
              {
                  'state_text': s.state_text,
                  'prompt': s.prompt,
                  'llm_response': s.llm_response,
                  'game_action': s.game_action_text,
                  'action_id': s.action_id,
                  'log_prob': s.log_prob,
              }
              for s in traj.steps
          ],
      }
      eval_record['players'].append(p_data)
    try:
      with open(eval_log_path, 'a') as f:
        f.write(json.dumps(eval_record) + '\n')
    except IOError as e:
      logging.warning('Failed to write eval episode log: %s', e)

  def evaluate(
      self,
      num_episodes: int = 10,
      eval_llm_max_horizon: int | None = None,
  ) -> dict[str, float]:
    """Evaluates the current policy over multiple episodes.

    If bot_partner is True and num_players == 2, splits episodes evenly
    across three conditions: P0=LLM vs P1=Bot, P0=Bot vs P1=LLM, and
    pure self-play (P0=LLM vs P1=LLM). Batches episodes in lockstep when
    eval_batch_size > 1 for faster inference.

    Args:
      num_episodes: Number of evaluation episodes.
      eval_llm_max_horizon: If set, LLM only generates moves up to this turn,
        and the heuristic bot plays the remainder of the game to terminal state.

    Returns:
      Dictionary of evaluation metrics.
    """
    self.backend.model.eval()
    num_players = self.game_config.num_players
    all_rewards = [[] for _ in range(num_players)]
    wins = np.zeros(num_players, dtype=np.int64)
    action_counts: list[dict[int, int]] = [{} for _ in range(num_players)]

    if self.bot_partner and num_players == 2:
      n_llm_bot = max(1, num_episodes // 3)
      n_bot_llm = max(1, num_episodes // 3)
      n_self_play = max(1, num_episodes - n_llm_bot - n_bot_llm)
      eval_plan = (
          [(1, 'P0:LLM vs P1:Bot')] * n_llm_bot
          + [(0, 'P0:Bot vs P1:LLM')] * n_bot_llm
          + [(None, 'Self-Play (LLM vs LLM)')] * n_self_play
      )
    else:
      eval_plan = [(None, 'Self-Play')] * num_episodes

    if eval_llm_max_horizon is not None:
      logging.info(
          '[evaluate] Running %d episodes: LLM plays turns [0, %d); heuristic'
          ' bot plays turns [%d, terminal).',
          len(eval_plan),
          eval_llm_max_horizon,
          eval_llm_max_horizon,
      )
    else:
      logging.info(
          '[evaluate] Running %d episodes: LLM plays all turns (no horizon handover).',
          len(eval_plan),
      )

    mode_rewards: dict[str, list[float]] = {}

    if self.eval_batch_size > 1 and hasattr(self.backend, 'generate_batch'):
      plan_chunks = [
          eval_plan[i : i + self.eval_batch_size]
          for i in range(0, len(eval_plan), self.eval_batch_size)
      ]
      ep_offset = 0
      for chunk in plan_chunks:
        batch_results = self.run_episodes_batch(
            chunk,
            is_evaluation=True,
            eval_llm_max_horizon=eval_llm_max_horizon,
        )
        for sub_i, (trajectories, state, mode_label) in enumerate(batch_results):
          self._process_eval_episode(
              ep_offset + sub_i,
              len(eval_plan),
              trajectories,
              state,
              mode_label,
              all_rewards,
              wins,
              action_counts,
              mode_rewards,
              num_players,
          )
        ep_offset += len(chunk)
    else:
      for ep_i, (bot_player, mode_label) in enumerate(eval_plan):
        trajectories = self.run_episode(
            is_evaluation=True,
            bot_player=bot_player,
            eval_llm_max_horizon=eval_llm_max_horizon,
        )
        state = self.env._state  # pylint: disable=protected-access
        self._process_eval_episode(
            ep_i,
            len(eval_plan),
            trajectories,
            state,
            mode_label,
            all_rewards,
            wins,
            action_counts,
            mode_rewards,
            num_players,
        )

    # Log action distribution summary across eval episodes.
    for p in range(num_players):
      total_actions = sum(action_counts[p].values())
      if total_actions > 0:
        dist_str = ', '.join(
            f'Action {a}: {count/total_actions*100:.0f}% ({count})'
            for a, count in sorted(action_counts[p].items())
        )
        logging.info('  Player %d eval action distribution: %s', p, dist_str)

    self.backend.model.train()

    metrics = {}
    for p in range(num_players):
      pr = np.array(all_rewards[p])
      metrics[f'eval/mean_reward_p{p}'] = float(np.mean(pr))
      metrics[f'eval/win_rate_p{p}'] = float(wins[p] / len(eval_plan))

    if self.bot_partner and num_players == 2:
      for m_label, r_list in mode_rewards.items():
        safe_key = m_label.lower().replace(' ', '_').replace(':', '_').replace('-', '_').replace('(', '').replace(')', '')
        metrics[f'eval/{safe_key}'] = float(np.mean(r_list)) if r_list else 0.0

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
    # If episode_rewards was not populated by run_episode (e.g. in GRPO),
    # record the step metrics directly so running averages work properly.
    if not self._episode_rewards or self._episode_rewards[-1] != mean_reward:
      self._episode_rewards.append(mean_reward)
      self._episode_losses.append(loss)

    window = min(len(self._episode_rewards), self.log_every)
    avg_r = (
        float(np.mean(self._episode_rewards[-window:]))
        if self._episode_rewards
        else mean_reward
    )
    avg_l = (
        float(np.mean(self._episode_losses[-window:]))
        if self._episode_losses
        else loss
    )
    logging.info(
        'Step %d | reward=%.4f (avg=%.4f) | loss=%.4f (avg=%.4f) | '
        '%.1f sec elapsed',
        ep,
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
    # Embed full experiment configuration for reproducibility.
    if self._experiment_config:
      summary['config'] = self._experiment_config
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

    # Update experiment config with the final GRPO config (may include
    # game-specific overrides, e.g. tiny_hanabi tuning).
    if self._experiment_config:
      import dataclasses as _dc  # pylint: disable=g-import-not-at-top
      grpo_fields = {
          f'grpo_{k}' if not k.startswith('grpo_') else k: v
          for k, v in _dc.asdict(grpo_config).items()
      }
      self._experiment_config.update(grpo_fields)
      config_path = os.path.join(self.results_dir, 'config.json')
      with open(config_path, 'w') as f:
        json.dump(self._experiment_config, f, indent=2)

    if getattr(grpo_config, 'bot_partner', False):
      self.bot_partner = True

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
        ref_state_dict=self._ref_state_dict,
    )
    runner.run()

    if self.use_wandb:
      import wandb  # pylint: disable=g-import-not-at-top

      wandb.finish()

