"""Trajectory dataclasses for RL training in TeamGamesRL.

These data structures record per-step and per-player trajectory information
collected during game episodes, used by REINFORCE and GRPO algorithms.
"""

import dataclasses


@dataclasses.dataclass
class RLTrajectoryStep:
  """A single decision step within a player's trajectory.

  Attributes:
    prompt: The full LLM prompt (game state + legal actions) shown to the
        model at this decision point.
    action_text: The action text selected by the model (stripped response).
    action_id: The integer OpenSpiel action ID.
    log_prob: The log-probability assigned by the LLM to the selected action.
    state_text: The rendered game state text (without action list).
    llm_response: The raw LLM response string.
    game_action_text: The OpenSpiel action-to-string representation.
  """
  prompt: str
  action_text: str
  action_id: int
  log_prob: float
  state_text: str = ''
  llm_response: str = ''
  game_action_text: str = ''


@dataclasses.dataclass
class PlayerTrajectory:
  """Stores the full trajectory for a single player within one episode.

  Attributes:
    player_id: Integer ID of the player this trajectory belongs to.
    steps: List of RLTrajectoryStep objects, one per decision point.
    reward: The episode return (final reward) for this player.
  """
  player_id: int
  steps: list[RLTrajectoryStep] = dataclasses.field(default_factory=list)
  reward: float = 0.0
