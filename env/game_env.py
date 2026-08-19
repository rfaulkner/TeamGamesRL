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

"""Environment creation and renderer factory for TeamGamesRL.

This module consolidates the duplicated environment-creation logic and
``_create_renderer()`` factory that were previously spread across
``gemma_rl_trainer.py`` and ``train.py``.
"""

from env import state_renderers
from env.game_config import GameConfig


def create_env(game_config: GameConfig):
  """Create an OpenSpiel RL environment from a game configuration.

  The ``rl_environment`` import is performed lazily (inside the function
  body) to mirror the original pattern and avoid importing heavy
  OpenSpiel modules at package-import time.

  Args:
    game_config: A ``GameConfig`` describing the game to instantiate.

  Returns:
    An ``open_spiel.python.rl_environment.Environment`` instance.
  """
  from open_spiel.python import rl_environment  # pylint: disable=g-import-not-at-top

  if game_config.game_params:
    return rl_environment.Environment(
        game_config.game_name, **game_config.game_params
    )
  return rl_environment.Environment(game_config.game_name)


def create_renderer(
    game_config: GameConfig,
) -> state_renderers.BaseStateRenderer:
  """Create a state renderer appropriate for the given game.

  Delegates to ``state_renderers.get_renderer`` which maps game names
  to concrete ``BaseStateRenderer`` subclasses.

  Args:
    game_config: A ``GameConfig`` specifying which game is being played.

  Returns:
    A ``BaseStateRenderer`` instance suitable for *game_config*.
  """
  return state_renderers.get_renderer(game_config.game_name)
