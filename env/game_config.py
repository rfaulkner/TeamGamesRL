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
