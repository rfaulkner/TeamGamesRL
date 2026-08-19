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

"""Game configuration registry for TeamGamesRL.

This module consolidates the ``GameConfig`` dataclass and the
``_GAME_CONFIGS`` registry that were previously duplicated across
``gemma_rl_trainer.py`` and ``train.py``.
"""

import dataclasses


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


# ---------------------------------------------------------------------------
# Named config constants
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_GAME_CONFIGS: dict[str, GameConfig] = {
    'negotiation': NEGOTIATION_CONFIG,
    'hanabi': HANABI_CONFIG,
    'tiny_hanabi': TINY_HANABI_CONFIG,
}

AVAILABLE_GAMES: list[str] = list(_GAME_CONFIGS.keys())


def get_game_config(name: str) -> GameConfig:
  """Look up a game configuration by name.

  Args:
    name: Registered game name (e.g. ``'hanabi'``).

  Returns:
    The corresponding ``GameConfig``.

  Raises:
    ValueError: If *name* is not a registered game.
  """
  if name not in _GAME_CONFIGS:
    raise ValueError(
        f'Unknown game {name!r}. Available games: {AVAILABLE_GAMES}'
    )
  return _GAME_CONFIGS[name]
