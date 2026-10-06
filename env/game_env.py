"""Environment creation and renderer factory for TeamGamesRL.

This module consolidates the duplicated environment-creation logic and
``_create_renderer()`` factory that were previously spread across
``gemma_rl_trainer.py`` and ``train.py``.
"""

from env import state_renderers
from env.game_config import GameConfig


def create_env(game_config: GameConfig, seed: int | None = None):
  """Create a game environment from a game configuration.

  For most games, this creates an OpenSpiel ``rl_environment.Environment``.
  For full Hanabi, it uses our HLE adapter (``env.hanabi.hanabi_env``)
  because OpenSpiel's Hanabi requires a custom C++ build.

  The ``rl_environment`` and HLE adapter imports are performed lazily
  to avoid importing heavy modules at package-import time.

  Args:
    game_config: A ``GameConfig`` describing the game to instantiate.
    seed: Optional seed for the environment's chance events (card deals).
      Two environments created with the same seed deal the same cards, which
      enables fixed-deal (paired) evaluation across training passes.  ``None``
      keeps the default, unseeded behaviour.

  Returns:
    An environment instance with ``reset()``, ``step()``, ``_state``,
    and ``game`` attributes.
  """
  if game_config.game_name == 'hanabi':
    from env.hanabi.hanabi_env import HanabiEnvironment  # pylint: disable=g-import-not-at-top
    from env.hanabi.hanabi_env import HanabiGame  # pylint: disable=g-import-not-at-top

    game = HanabiGame(**game_config.game_params, seed=seed)
    return HanabiEnvironment(game)

  try:
    from open_spiel.python import rl_environment  # pylint: disable=g-import-not-at-top
  except ImportError:
    try:
      from third_party.open_spiel.python import rl_environment  # pylint: disable=g-import-not-at-top
    except ImportError:
      from google3.third_party.open_spiel.python import rl_environment  # pylint: disable=g-import-not-at-top

  env_kwargs = dict(game_config.game_params or {})
  if seed is not None:
    env_kwargs['chance_event_sampler'] = rl_environment.ChanceEventSampler(
        seed=seed
    )
  if env_kwargs:
    return rl_environment.Environment(game_config.game_name, **env_kwargs)
  return rl_environment.Environment(game_config.game_name)


def create_renderer(
    game_config: GameConfig,
    max_history_turns: int | None = 20,
) -> state_renderers.BaseStateRenderer:
  """Create a state renderer appropriate for the given game.

  Delegates to ``state_renderers.get_renderer`` which maps game names
  to concrete ``BaseStateRenderer`` subclasses.

  Args:
    game_config: A ``GameConfig`` specifying which game is being played.
    max_history_turns: For Hanabi, the maximum number of recent moves
      to include in the prompt. ``None`` or ``0`` shows all moves.

  Returns:
    A ``BaseStateRenderer`` instance suitable for *game_config*.
  """
  return state_renderers.get_renderer(
      game_config.game_name,
      max_history_turns=max_history_turns,
  )
