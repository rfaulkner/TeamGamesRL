# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main training loop for multi-agent LLM RL on OpenSpiel games.

This module implements a REINFORCE-based training loop where LLM agents play
episodes of OpenSpiel games and update their behavior based on game rewards.

Architecture overview:
  1. An OpenSpiel game is loaded and wrapped in an RL environment.
  2. Each player is controlled by an LLMAgent that uses a language model
     (or a mock LLM for testing) to select actions from text-rendered game
     states.
  3. Episodes are played by stepping through the environment, collecting
     trajectories of (state, action, log_prob) tuples for each player.
  4. After each episode, the REINFORCE policy gradient loss is computed:
       loss = -sum(log_prob_t * R)  for each step t in the trajectory,
     where R is the episode return for the corresponding player.
  5. Metrics (rewards, win rates, episode lengths, losses) are logged
     periodically.

The training loop is designed to be modular:
  - State renderers convert raw OpenSpiel states to text prompts.
  - LLM agents produce action distributions from text prompts.
  - The trainer orchestrates episode collection and loss computation.

Note: Actual gradient-based weight updates on LLM parameters require a
differentiable training framework (e.g., JAX/PyTorch). This module computes
the loss values and logs them. The weight-update mechanism will be connected
when a real trainable model backend is integrated.

Usage:
  python train.py --game=tiny_hanabi --num_episodes=100 --eval_every=10

  # With a specific LLM type (default is mock for testing):
  python train.py --game=negotiation --llm_type=mock --temperature=0.8
"""

import dataclasses
import os
import time

from absl import app
from absl import flags
from absl import logging
import numpy as np

from open_spiel.python import rl_environment

import llm_agent
import state_renderers

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    'game', 'tiny_hanabi', ['negotiation', 'hanabi', 'tiny_hanabi'],
    'Name of the OpenSpiel game to train on.')
flags.DEFINE_integer(
    'num_episodes', 1000,
    'Total number of training episodes to run.')
flags.DEFINE_integer(
    'eval_every', 100,
    'Run evaluation every this many episodes.')
flags.DEFINE_integer(
    'num_eval_episodes', 20,
    'Number of episodes to run during each evaluation.')
flags.DEFINE_float(
    'temperature', 1.0,
    'Sampling temperature for LLM action selection.')
flags.DEFINE_enum(
    'llm_type', 'mock', ['mock', 'gemini'],
    'Type of LLM backend to use. "mock" uses a random policy for testing.')
flags.DEFINE_string(
    'output_dir', '/tmp/llm_rl_logs',
    'Directory for writing training logs and checkpoints.')
flags.DEFINE_integer(
    'log_every', 10,
    'Log training metrics every this many episodes.')
flags.DEFINE_integer(
    'seed', 42,
    'Random seed for reproducibility.')


# ============================================================================
# Game Configurations
# ============================================================================


@dataclasses.dataclass(frozen=True)
class GameConfig:
  """Configuration for an OpenSpiel game.

  Attributes:
    game_name: The OpenSpiel registered game name string.
    game_params: Dictionary of game-specific parameters passed to the
      OpenSpiel game constructor.
    num_players: Number of players in the game.
  """
  game_name: str
  game_params: dict[str, object]
  num_players: int


NEGOTIATION_CONFIG = GameConfig(
    game_name='negotiation',
    game_params={},
    num_players=2,
)

HANABI_CONFIG = GameConfig(
    game_name='hanabi',
    game_params={
        'players': 2,
    },
    num_players=2,
)

TINY_HANABI_CONFIG = GameConfig(
    game_name='tiny_hanabi',
    game_params={},
    num_players=2,
)

_GAME_CONFIGS = {
    'negotiation': NEGOTIATION_CONFIG,
    'hanabi': HANABI_CONFIG,
    'tiny_hanabi': TINY_HANABI_CONFIG,
}


# ============================================================================
# Trajectory Data
# ============================================================================


@dataclasses.dataclass
class Trajectory:
  """Stores the trajectory data for a single player within one episode.

  A trajectory records the sequence of observations, actions, and
  associated log-probabilities for one player, along with the final
  reward received at the end of the episode.

  Attributes:
    player_id: Integer ID of the player this trajectory belongs to.
    states: List of text prompts (rendered game states) seen by the
      player at each decision point.
    actions: List of integer action IDs chosen by the player.
    action_texts: List of human-readable text descriptions of the
      actions taken.
    log_probs: List of log-probabilities assigned by the LLM to each
      chosen action.
    reward: The episode return (final reward) for this player.
  """
  player_id: int
  states: list[str] = dataclasses.field(default_factory=list)
  actions: list[int] = dataclasses.field(default_factory=list)
  action_texts: list[str] = dataclasses.field(default_factory=list)
  log_probs: list[float] = dataclasses.field(default_factory=list)
  reward: float = 0.0

  @property
  def num_steps(self) -> int:
    """Returns the number of decision steps in this trajectory."""
    return len(self.actions)


# ============================================================================
# RL Trainer
# ============================================================================


class RLTrainer:
  """Orchestrates REINFORCE-based training of LLM agents on OpenSpiel games.

  The trainer manages the interaction loop between LLM agents and an
  OpenSpiel environment, collects per-player trajectories, computes
  REINFORCE policy gradient loss estimates, and logs training metrics.

  Typical usage:
    ```
    trainer = RLTrainer(
        game_name='tiny_hanabi',
        agents=[agent_0, agent_1],
        renderers=[renderer_0, renderer_1],
        num_episodes=1000,
        eval_every=100,
        log_dir='/tmp/llm_rl_logs',
    )
    trainer.train()
    ```

  Attributes:
    game_config: The GameConfig for the selected game.
    env: The OpenSpiel RL environment instance.
    agents: List of LLMAgent instances, one per player.
    renderers: List of StateRenderer instances, one per player.
    num_episodes: Total number of training episodes.
    eval_every: Frequency (in episodes) of evaluation runs.
  """

  def __init__(
      self,
      game_name: str,
      env: rl_environment.Environment,
      agents: list[object],
      renderers: list['state_renderers.BaseStateRenderer'],
      num_episodes: int = 1000,
      eval_every: int = 100,
      log_dir: str = '/tmp/llm_rl_logs',
  ):
    """Initializes the RLTrainer.

    Args:
      game_name: Key into _GAME_CONFIGS specifying which game to play.
      env: The rl_environment.Environment instance to train on.
      agents: List of LLMAgent instances, one per player. Each agent
        must implement a `step(time_step, is_evaluation)` method that
        returns a StepOutput with `action`, `probs`, and `log_prob`.
      renderers: List of StateRenderer instances, one per player. Each
        renderer converts raw OpenSpiel states to text prompts.
      num_episodes: Total number of training episodes to run.
      eval_every: Run evaluation every this many episodes.
      log_dir: Directory path for storing training logs.

    Raises:
      ValueError: If game_name is not a recognized game configuration,
        or if the number of agents/renderers does not match the game's
        player count.
    """
    if game_name not in _GAME_CONFIGS:
      raise ValueError(
          f'Unknown game: {game_name}. '
          f'Available games: {list(_GAME_CONFIGS.keys())}')

    self.game_config = _GAME_CONFIGS[game_name]
    self.num_episodes = num_episodes
    self.eval_every = eval_every
    self.log_dir = log_dir

    num_players = self.game_config.num_players
    if len(agents) != num_players:
      raise ValueError(
          f'Expected {num_players} agents for game {game_name}, '
          f'got {len(agents)}.')
    if len(renderers) != num_players:
      raise ValueError(
          f'Expected {num_players} renderers for game {game_name}, '
          f'got {len(renderers)}.')

    self.agents = agents
    self.renderers = renderers
    self.env = env

    # Training metrics accumulators.
    self._episode_rewards = []  # Per-episode mean reward across players.
    self._episode_lengths = []  # Per-episode step counts.
    self._episode_losses = []  # Per-episode REINFORCE loss values.
    self._player_wins = np.zeros(num_players, dtype=np.int64)
    self._total_episodes = 0

    logging.info(
        'RLTrainer initialized: game=%s, num_players=%d, '
        'num_episodes=%d, eval_every=%d',
        game_name, num_players, num_episodes, eval_every)

  def run_episode(self, is_evaluation: bool = False) -> list[Trajectory]:
    """Plays one full episode and returns per-player trajectories.

    Steps through the OpenSpiel environment from initial state to
    terminal state, collecting trajectory data for each player at
    every decision point.

    The episode loop:
      1. Reset the environment to get the first TimeStep.
      2. While the episode is not over:
         a. Determine the current player.
         b. Render the game state as text for the current player.
         c. Call the agent's step() method to get an action.
         d. Record the state text, action, action text, and log-prob
            in the player's trajectory.
         e. Step the environment with the chosen action.
      3. After the episode ends, assign final rewards to each
         player's trajectory from the terminal TimeStep.

    Args:
      is_evaluation: If True, agents may use evaluation-mode behavior
        (e.g., greedy instead of sampled actions).

    Returns:
      A list of Trajectory objects, one per player, containing the
      full episode data.
    """
    num_players = self.game_config.num_players
    trajectories = [
        Trajectory(player_id=p) for p in range(num_players)
    ]

    time_step = self.env.reset()
    episode_length = 0

    while not time_step.last():
      current_player = time_step.current_player()

      # Render the current state as text for the acting player.
      state = self.env._state
      state_text = self.renderers[current_player].render_state(
          state, current_player, self.env.game)

      # Get action from the agent.
      step_output = self.agents[current_player].step(
          time_step, is_evaluation=is_evaluation)
      action = step_output.action

      # Get action text description from the game state.
      action_text = self.env.get_state.action_to_string(
          current_player, action)

      # Extract log-probability if available, otherwise compute from probs.
      if hasattr(step_output, 'log_prob') and step_output.log_prob is not None:
        log_prob = step_output.log_prob
      elif step_output.probs is not None and len(step_output.probs) > 0:
        prob = step_output.probs[action]
        log_prob = float(np.log(max(prob, 1e-10)))
      else:
        log_prob = 0.0

      # Record trajectory data for this player.
      traj = trajectories[current_player]
      traj.states.append(state_text)
      traj.actions.append(action)
      traj.action_texts.append(action_text)
      traj.log_probs.append(log_prob)

      # Step the environment.
      time_step = self.env.step([action])
      episode_length += 1

    # Assign final rewards from the terminal TimeStep.
    if time_step.rewards is not None:
      for p in range(num_players):
        trajectories[p].reward = time_step.rewards[p]

    return trajectories

  def compute_reinforce_loss(self, trajectories: list[Trajectory]) -> float:
    """Computes the REINFORCE policy gradient loss for a set of trajectories.

    The REINFORCE loss for a single trajectory is:
      L = -sum_t( log_prob(a_t | s_t) * R )
    where R is the episode return for the player. The total loss is the
    sum across all players' trajectories.

    This loss value represents what would be minimized via gradient
    descent to increase the probability of actions that led to higher
    rewards. The actual gradient computation and weight update require
    a differentiable training framework and will be added when a real
    trainable model backend is integrated.

    Args:
      trajectories: List of Trajectory objects from a single episode.

    Returns:
      The scalar REINFORCE loss value (float). A lower (more negative)
      loss indicates the policy is being reinforced toward high-reward
      actions.
    """
    total_loss = 0.0
    for traj in trajectories:
      if traj.num_steps == 0:
        continue
      # REINFORCE: loss = -sum(log_prob_t * reward)
      for log_prob in traj.log_probs:
        total_loss += -log_prob * traj.reward
    return total_loss

  def evaluate(self, num_eval_episodes: int = 20) -> dict[str, float]:
    """Evaluates the current agents over multiple episodes.

    Runs episodes in evaluation mode (agents may behave differently,
    e.g., using greedy action selection) and collects aggregate
    performance metrics.

    Args:
      num_eval_episodes: Number of evaluation episodes to run.

    Returns:
      A dictionary containing evaluation metrics:
        - 'mean_reward_player_{i}': Mean reward for player i.
        - 'std_reward_player_{i}': Std dev of reward for player i.
        - 'win_rate_player_{i}': Fraction of episodes won by player i.
        - 'mean_episode_length': Mean number of steps per episode.
        - 'mean_loss': Mean REINFORCE loss across episodes.
    """
    num_players = self.game_config.num_players
    all_rewards = [[] for _ in range(num_players)]
    episode_lengths = []
    losses = []
    wins = np.zeros(num_players, dtype=np.int64)

    for _ in range(num_eval_episodes):
      trajectories = self.run_episode(is_evaluation=True)

      ep_length = sum(t.num_steps for t in trajectories)
      episode_lengths.append(ep_length)

      loss = self.compute_reinforce_loss(trajectories)
      losses.append(loss)

      # Collect per-player rewards and determine winner.
      rewards = [t.reward for t in trajectories]
      for p in range(num_players):
        all_rewards[p].append(rewards[p])

      # The winner is the player with the highest reward (if unique).
      max_reward = max(rewards)
      winners = [p for p in range(num_players) if rewards[p] == max_reward]
      if len(winners) == 1:
        wins[winners[0]] += 1

    # Compile metrics.
    metrics = {}
    for p in range(num_players):
      player_rewards = np.array(all_rewards[p])
      metrics[f'mean_reward_player_{p}'] = float(np.mean(player_rewards))
      metrics[f'std_reward_player_{p}'] = float(np.std(player_rewards))
      metrics[f'win_rate_player_{p}'] = float(wins[p] / num_eval_episodes)

    metrics['mean_episode_length'] = float(np.mean(episode_lengths))
    metrics['mean_loss'] = float(np.mean(losses))

    return metrics

  def train(self) -> None:
    """Runs the main REINFORCE training loop.

    For each episode:
      1. Play an episode, collecting per-player trajectories.
      2. Compute the REINFORCE loss.
      3. Log training metrics (rewards, losses, episode lengths).
      4. Periodically run evaluation and log evaluation metrics.

    Note: This loop computes loss values but does not perform gradient
    updates on model weights. Weight updates will be added when a real
    trainable LLM backend is integrated. With MockLLM, this validates
    the full training pipeline end-to-end.
    """
    logging.info('Starting training for %d episodes.', self.num_episodes)
    start_time = time.time()
    log_every = FLAGS['log_every'].value if FLAGS['log_every'].present else 10

    # Rolling window for recent metrics.
    recent_window = min(100, max(10, self.num_episodes // 10))
    recent_rewards = []
    recent_lengths = []
    recent_losses = []

    for episode_idx in range(1, self.num_episodes + 1):
      # Run one training episode.
      trajectories = self.run_episode(is_evaluation=False)

      # Compute REINFORCE loss.
      loss = self.compute_reinforce_loss(trajectories)

      # Collect episode metrics.
      episode_rewards = [t.reward for t in trajectories]
      mean_reward = float(np.mean(episode_rewards))
      episode_length = sum(t.num_steps for t in trajectories)

      # Update accumulators.
      self._episode_rewards.append(mean_reward)
      self._episode_lengths.append(episode_length)
      self._episode_losses.append(loss)
      self._total_episodes += 1

      # Track wins.
      max_reward = max(episode_rewards)
      winners = [
          p for p in range(self.game_config.num_players)
          if episode_rewards[p] == max_reward
      ]
      if len(winners) == 1:
        self._player_wins[winners[0]] += 1

      # Update rolling window.
      recent_rewards.append(mean_reward)
      recent_lengths.append(episode_length)
      recent_losses.append(loss)
      if len(recent_rewards) > recent_window:
        recent_rewards.pop(0)
        recent_lengths.pop(0)
        recent_losses.pop(0)

      # Periodic logging.
      if episode_idx % log_every == 0:
        avg_reward = float(np.mean(recent_rewards))
        avg_length = float(np.mean(recent_lengths))
        avg_loss = float(np.mean(recent_losses))
        elapsed = time.time() - start_time
        eps_per_sec = episode_idx / elapsed if elapsed > 0 else 0.0

        logging.info(
            'Episode %d/%d | reward=%.4f (avg=%.4f) | '
            'length=%d (avg=%.1f) | loss=%.4f (avg=%.4f) | '
            '%.1f eps/sec',
            episode_idx, self.num_episodes,
            mean_reward, avg_reward,
            episode_length, avg_length,
            loss, avg_loss,
            eps_per_sec)

        # Log per-player rewards.
        for p in range(self.game_config.num_players):
          logging.info(
              '  Player %d: reward=%.4f, wins=%d/%d (%.1f%%)',
              p, episode_rewards[p],
              self._player_wins[p], self._total_episodes,
              100.0 * self._player_wins[p] / self._total_episodes)

      # Periodic evaluation.
      if episode_idx % self.eval_every == 0:
        logging.info('--- Evaluation at episode %d ---', episode_idx)
        eval_metrics = self.evaluate(
            num_eval_episodes=FLAGS['num_eval_episodes'].value
            if FLAGS['num_eval_episodes'].present else 20)

        for key, value in sorted(eval_metrics.items()):
          logging.info('  eval/%s: %.4f', key, value)
        logging.info('--- End evaluation ---')

    # Final summary.
    total_time = time.time() - start_time
    logging.info('Training complete: %d episodes in %.1f seconds.',
                 self.num_episodes, total_time)
    logging.info('Final metrics:')
    logging.info('  Mean reward (all episodes): %.4f',
                 float(np.mean(self._episode_rewards)))
    logging.info('  Mean episode length: %.1f',
                 float(np.mean(self._episode_lengths)))
    logging.info('  Mean REINFORCE loss: %.4f',
                 float(np.mean(self._episode_losses)))
    for p in range(self.game_config.num_players):
      logging.info('  Player %d win rate: %.1f%% (%d/%d)',
                   p,
                   100.0 * self._player_wins[p] / self._total_episodes,
                   self._player_wins[p], self._total_episodes)


# ============================================================================
# Entry Point
# ============================================================================


def _create_renderer(game_config: GameConfig, player_id: int) -> object:
  """Creates a state renderer appropriate for the given game and player.

  Args:
    game_config: The GameConfig specifying which game is being played.
    player_id: The player ID this renderer is for.

  Returns:
    A StateRenderer instance.
  """
  game_name = game_config.game_name
  if game_name == 'tiny_hanabi':
    return state_renderers.TinyHanabiRenderer()
  elif game_name == 'hanabi':
    return state_renderers.HanabiRenderer()
  elif game_name == 'negotiation':
    return state_renderers.NegotiationRenderer()
  else:
    # Fallback to a generic renderer that uses OpenSpiel's string
    # representation.
    return state_renderers.GenericRenderer()


def _create_agent(
    game_config: GameConfig,
    player_id: int,
    llm_type: str,
    temperature: float,
    renderer,
    env,
) -> object:
  """Creates an LLM agent for the given game and player.

  Args:
    game_config: The GameConfig specifying which game is being played.
    player_id: The player ID this agent controls.
    llm_type: The LLM backend type ('mock' or 'gemini').
    temperature: Sampling temperature for action selection.
    renderer: The StateRenderer for this agent.
    env: The OpenSpiel rl_environment.Environment object.

  Returns:
    An LLMAgent instance.

  Raises:
    ValueError: If llm_type is not recognized.
  """
  if llm_type == 'mock':
    llm_backend = llm_agent.MockLLM()
  elif llm_type == 'gemini':
    llm_backend = llm_agent.GeminiLLM(temperature=temperature)
  else:
    raise ValueError(f'Unknown LLM type: {llm_type}. Use "mock" or "gemini".')

  return llm_agent.LLMAgent(
      player_id=player_id,
      renderer=renderer,
      llm=llm_backend,
      env=env,
      temperature=temperature,
  )


def main(argv: list[str]) -> None:
  """Main entry point for the LLM RL training script.

  Parses flags, creates the game environment, agents, and renderers,
  then launches the training loop.

  Args:
    argv: Command-line arguments (unused beyond flag parsing).
  """
  del argv  # Unused.

  # Set random seed for reproducibility.
  np.random.seed(FLAGS.seed)

  game_name = FLAGS.game
  game_config = _GAME_CONFIGS[game_name]

  logging.info('=== LLM RL Training ===')
  logging.info('Game: %s (%s)', game_name, game_config.game_name)
  logging.info('LLM type: %s', FLAGS.llm_type)
  logging.info('Temperature: %.2f', FLAGS.temperature)
  logging.info('Episodes: %d', FLAGS.num_episodes)
  logging.info('Eval every: %d episodes', FLAGS.eval_every)
  logging.info('Log dir: %s', FLAGS.output_dir)

  # Create a temporary environment for spec extraction.
  if game_config.game_params:
    env = rl_environment.Environment(
        game_config.game_name, **game_config.game_params)
  else:
    env = rl_environment.Environment(game_config.game_name)

  # Create per-player renderers and agents.
  renderers = []
  agents = []
  for player_id in range(game_config.num_players):
    renderers.append(_create_renderer(game_config, player_id))
    agents.append(_create_agent(
        game_config, player_id, FLAGS.llm_type, FLAGS.temperature, renderers[-1], env))

  # Create log directory.
  os.makedirs(FLAGS.output_dir, exist_ok=True)
  log_path = os.path.join(FLAGS.output_dir, f'train_{game_name}.jsonl')
  # Create the trainer and run.
  trainer = RLTrainer(
      game_name=game_name,
      env=env,
      agents=agents,
      renderers=renderers,
      num_episodes=FLAGS.num_episodes,
      eval_every=FLAGS.eval_every,
      log_dir=FLAGS.output_dir,
  )
  trainer.train()


if __name__ == '__main__':
  app.run(main)
