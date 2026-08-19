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
