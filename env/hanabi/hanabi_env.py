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

"""OpenSpiel-compatible adapter for the Hanabi Learning Environment (HLE).

This module provides ``HanabiState``, ``HanabiGame``, and
``HanabiEnvironment`` classes that wrap the standalone
``hanabi_learning_environment`` package and present the same interface
consumed by our training pipeline (``grpo_sampled.py``,
``HanabiRenderer``, ``rl_trainer.py``).

Why this adapter?
  OpenSpiel's C++ Hanabi game (``pyspiel.load_game('hanabi')``) requires
  building OpenSpiel from source with ``-DBUILD_WITH_HANABI=ON``, which
  is fragile on HPC clusters.  This adapter uses the standalone HLE
  package (``pip install hanabi-learning-environment``) instead.

Usage::

  from env.hanabi.hanabi_env import HanabiEnvironment, HanabiGame

  game = HanabiGame(players=2)
  env = HanabiEnvironment(game)
  time_step = env.reset()
  while not time_step.last():
      state = env._state
      # ... same API as OpenSpiel ...
      time_step = env.step([action_id])
"""

from __future__ import annotations

import json

from hanabi_learning_environment import pyhanabi


# ── Color helpers ──────────────────────────────────────────────────────────

_COLOR_CHARS = ['R', 'Y', 'G', 'W', 'B']  # matches OpenSpiel ordering


def _color_char(color_idx: int) -> str:
  """Map HLE color index → single-char color code (R/Y/G/W/B)."""
  if 0 <= color_idx < len(_COLOR_CHARS):
    return _COLOR_CHARS[color_idx]
  return '?'


# ═══════════════════════════════════════════════════════════════════════════
# HanabiState — wraps pyhanabi.HanabiState
# ═══════════════════════════════════════════════════════════════════════════


class HanabiState:
  """OpenSpiel-compatible state wrapper around HLE's pyhanabi state.

  Provides the same method signatures that ``HanabiRenderer``,
  ``collect_game_prompts``, and ``simulate_from_state`` call on an
  OpenSpiel state.
  """

  def __init__(
      self, game: HanabiGame, hle_state: pyhanabi.HanabiState
  ) -> None:
    self._game = game
    self._hle_state = hle_state
    self._action_history: list[int] = []

  # ── Core queries ───────────────────────────────────────────────────────

  def current_player(self) -> int:
    """Return the index of the player whose turn it is."""
    return self._hle_state.cur_player()

  def is_terminal(self) -> bool:
    """Return True when the game is over."""
    return self._hle_state.is_terminal()

  def is_chance_node(self) -> bool:
    """HLE handles chance internally, so this is always False."""
    return self._hle_state.cur_player() == pyhanabi.CHANCE_PLAYER_ID

  def returns(self) -> list[float]:
    """Return per-player cumulative returns (same for all in co-op)."""
    score = float(self._hle_state.score())
    return [score] * self._game.num_players()

  def rewards(self) -> list[float]:
    """Alias for returns() in cooperative games."""
    return self.returns()

  # ── Actions ────────────────────────────────────────────────────────────

  def legal_actions(self, player: int | None = None) -> list[int]:
    """Return list of legal action UIDs for the current player."""
    if player is not None and player != self.current_player():
      return []
    return self._hle_state.legal_moves_as_int()

  def action_to_string(self, player: int, action_id: int) -> str:
    """Convert an action UID to a human-readable string.

    Matches the OpenSpiel format:
      "(Play 0)", "(Discard 2)", "(Reveal player +1 color R)",
      "(Reveal player +1 rank 3)"
    """
    move = self._game._hle_game.get_move(action_id)
    return _move_to_string(move, player, self._game.num_players())

  def apply_action(self, action_id: int) -> None:
    """Apply an action and advance the game state."""
    move = self._game._hle_game.get_move(action_id)
    self._hle_state.apply_move(move)
    self._action_history.append(action_id)

  # ── Observations ───────────────────────────────────────────────────────

  def observation_string(self, player: int | None = None) -> str:
    """Return the observation string for ``player``.

    Reproduces the OpenSpiel Hanabi observation format that
    ``HanabiRenderer._parse_hanabi_observation()`` expects::

      Life tokens: 3
      Info tokens: 8
      Fireworks: R0 Y0 G0 W0 B0
      Hands:
      Cur player
      XX || XX|RYGWB12345
      ...
      -----
      R2 || R2|RYGWB12345
      ...
      Deck size: 40
      Discards:
    """
    if player is None:
      player = self.current_player()
    obs = pyhanabi.HanabiObservation(
        self._hle_state, self._hle_state.observation(player)
    )
    return _format_observation(
        obs, player, self._game._hle_game, self._hle_state
    )

  # ── Serialization ──────────────────────────────────────────────────────

  def serialize(self) -> str:
    """Serialize the state to a JSON string for save/restore."""
    return json.dumps({
        'action_history': self._action_history,
    })

  @classmethod
  def deserialize(cls, game: HanabiGame, data_str: str) -> HanabiState:
    """Restore a state by replaying its action history."""
    data = json.loads(data_str)
    state = game.new_initial_state()
    for action_id in data['action_history']:
      if state.is_terminal():
        break
      state.apply_action(action_id)
    return state

  def clone(self) -> HanabiState:
    """Deep-copy this state."""
    new_state = HanabiState(self._game, self._hle_state.copy())
    new_state._action_history = list(self._action_history)
    return new_state


# ═══════════════════════════════════════════════════════════════════════════
# HanabiGame — wraps pyhanabi.HanabiGame
# ═══════════════════════════════════════════════════════════════════════════


class HanabiGame:
  """OpenSpiel-compatible game object wrapping HLE.

  Provides the ``new_initial_state()``, ``deserialize_state()``, and
  metadata methods that the training pipeline expects.
  """

  def __init__(
      self,
      players: int = 2,
      colors: int = 5,
      ranks: int = 5,
      hand_size: int = 5,
      max_information_tokens: int = 8,
      max_life_tokens: int = 3,
      random_start_player: bool = False,
  ) -> None:
    params = {
        'players': players,
        'colors': colors,
        'rank': ranks,
        'hand_size': hand_size,
        'max_information_tokens': max_information_tokens,
        'max_life_tokens': max_life_tokens,
        'random_start_player': random_start_player,
    }
    self._hle_game = pyhanabi.HanabiGame(params)
    self._params = params

  def num_players(self) -> int:
    return self._params['players']

  def new_initial_state(self) -> HanabiState:
    """Create a new game state (deals cards randomly)."""
    hle_state = self._hle_game.new_initial_state()
    # HLE deals cards via chance nodes — advance past them.
    while hle_state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
      hle_state.deal_random_card()
    return HanabiState(self, hle_state)

  def deserialize_state(self, data_str: str) -> HanabiState:
    """Restore a state from its serialized form."""
    return HanabiState.deserialize(self, data_str)


# ═══════════════════════════════════════════════════════════════════════════
# HanabiEnvironment — wraps rl_env and presents OpenSpiel-like API
# ═══════════════════════════════════════════════════════════════════════════


class _TimeStep:
  """Minimal OpenSpiel-compatible TimeStep."""

  def __init__(
      self,
      current_player: int,
      is_last: bool,
      rewards: list[float] | None = None,
  ) -> None:
    self._current_player = current_player
    self._is_last = is_last
    self._rewards = rewards

  def current_player(self) -> int:
    return self._current_player

  def last(self) -> bool:
    return self._is_last

  def rewards(self) -> list[float] | None:
    return self._rewards


class HanabiEnvironment:
  """OpenSpiel ``rl_environment.Environment``-compatible wrapper for HLE.

  Provides ``reset()``, ``step()``, ``_state``, ``game``, and
  ``set_state()`` with the same semantics as OpenSpiel's
  ``rl_environment.Environment``.
  """

  def __init__(self, hanabi_game: HanabiGame) -> None:
    self.game = hanabi_game
    self._state: HanabiState | None = None

  def reset(self) -> _TimeStep:
    """Start a new episode and return the initial TimeStep."""
    self._state = self.game.new_initial_state()
    return _TimeStep(
        current_player=self._state.current_player(),
        is_last=self._state.is_terminal(),
    )

  def step(self, actions: list[int]) -> _TimeStep:
    """Apply an action and return the new TimeStep.

    Args:
      actions: A list with a single action ID (matches OpenSpiel's
        multi-agent step convention where only the current player's
        action slot matters).

    Returns:
      A ``_TimeStep`` with updated current_player and terminal status.
    """
    if self._state is None:
      raise RuntimeError('Must call reset() before step().')

    action_id = actions[0] if isinstance(actions, list) else actions
    self._state.apply_action(action_id)

    if self._state.is_terminal():
      return _TimeStep(
          current_player=-1,
          is_last=True,
          rewards=self._state.returns(),
      )
    return _TimeStep(
        current_player=self._state.current_player(),
        is_last=False,
    )

  def set_state(self, state: HanabiState) -> None:
    """Replace the current state (used by simulate_from_state)."""
    self._state = state


# ═══════════════════════════════════════════════════════════════════════════
# Serialization helpers compatible with grpo_sampled.py
# ═══════════════════════════════════════════════════════════════════════════


def serialize_game_and_state(
    game: HanabiGame, state: HanabiState
) -> str:
  """Serialize a game+state pair for later restoration.

  Replaces ``pyspiel.serialize_game_and_state()`` for our adapter.
  """
  return json.dumps({
      'adapter': 'hanabi_env',
      'params': game._params,
      'state': json.loads(state.serialize()),
  })


def deserialize_game_and_state(data_str: str) -> tuple[HanabiGame, HanabiState]:
  """Restore a game+state pair.

  Replaces ``pyspiel.deserialize_game_and_state()`` for our adapter.
  """
  data = json.loads(data_str)
  game = HanabiGame(**data['params'])
  state = game.new_initial_state()
  for action_id in data['state']['action_history']:
    if state.is_terminal():
      break
    state.apply_action(action_id)
  return game, state


# ═══════════════════════════════════════════════════════════════════════════
# Internal formatting helpers
# ═══════════════════════════════════════════════════════════════════════════


def _move_to_string(
    move: pyhanabi.HanabiMove, player: int, num_players: int
) -> str:
  """Convert an HLE move to the OpenSpiel action string format."""
  move_type = move.type()

  if move_type == pyhanabi.HanabiMove.Type.PLAY:
    return f'(Play {move.card_index()})'
  elif move_type == pyhanabi.HanabiMove.Type.DISCARD:
    return f'(Discard {move.card_index()})'
  elif move_type == pyhanabi.HanabiMove.Type.REVEAL_COLOR:
    target = move.target_offset()
    color = _color_char(move.color())
    return f'(Reveal player +{target} color {color})'
  elif move_type == pyhanabi.HanabiMove.Type.REVEAL_RANK:
    target = move.target_offset()
    rank = move.rank() + 1  # HLE is 0-indexed, OpenSpiel shows 1-indexed
    return f'(Reveal player +{target} rank {rank})'
  else:
    return f'(Unknown move type {move_type})'


def _format_observation(
    obs: pyhanabi.HanabiObservation,
    player: int,
    hle_game: pyhanabi.HanabiGame,
    hle_state: pyhanabi.HanabiState,
) -> str:
  """Format an HLE observation to match OpenSpiel's observation_string.

  Produces the format ``HanabiRenderer._parse_hanabi_observation()``
  expects, with sections: Life tokens, Info tokens, Fireworks, Hands,
  Deck size, Discards.
  """
  lines = []

  # Life and info tokens.
  lines.append(f'Life tokens: {obs.life_tokens()}')
  lines.append(f'Info tokens: {obs.information_tokens()}')

  # Fireworks (completed stacks per color).
  fireworks = obs.fireworks()
  fw_parts = []
  for color_idx, score in enumerate(fireworks):
    fw_parts.append(f'{_color_char(color_idx)}{score}')
  lines.append(f'Fireworks: {" ".join(fw_parts)}')

  # Hands.
  lines.append('Hands:')
  num_players = hle_game.num_players()
  num_colors = hle_game.num_colors()
  num_ranks = hle_game.num_ranks()

  for offset in range(num_players):
    pid = (player + offset) % num_players
    if pid == player:
      # Current player's hand — show "XX" for card face, knowledge only.
      lines.append('Cur player')
      hand = obs.card_knowledge()[0]  # own hand is always index 0
      for card_knowledge in hand:
        knowledge_str = _format_card_knowledge(
            card_knowledge, num_colors, num_ranks
        )
        lines.append(f'XX || {knowledge_str}')
    else:
      # Other player's hand — show actual cards.
      # Observed hands are indexed 1..N-1 for the other players.
      observed_idx = offset - 1 if offset > 0 else 0
      observed_hands = obs.observed_hands()
      if observed_idx < len(observed_hands):
        hand_cards = observed_hands[observed_idx]
        hand_knowledge = obs.card_knowledge()[offset]
        for card_idx, card in enumerate(hand_cards):
          card_str = f'{_color_char(card.color())}{card.rank() + 1}'
          if card_idx < len(hand_knowledge):
            knowledge_str = _format_card_knowledge(
                hand_knowledge[card_idx], num_colors, num_ranks
            )
          else:
            knowledge_str = 'XX'
          lines.append(f'{card_str} || {knowledge_str}')
    lines.append('-----')

  # Deck size.
  lines.append(f'Deck size: {obs.deck_size()}')

  # Discards.
  discard_pile = obs.discard_pile()
  if discard_pile:
    discard_strs = [
        f'{_color_char(c.color())}{c.rank() + 1}' for c in discard_pile
    ]
    lines.append(f'Discards: {" ".join(discard_strs)}')
  else:
    lines.append('Discards:')

  return '\n'.join(lines)


def _format_card_knowledge(
    knowledge: pyhanabi.HanabiCardKnowledge,
    num_colors: int,
    num_ranks: int,
) -> str:
  """Format card knowledge to match OpenSpiel's format.

  OpenSpiel format: "XX|RYGWB12345" where known info is narrowed.
  """
  # Color knowledge.
  if knowledge.color() is not None:
    color_str = _color_char(knowledge.color())
  else:
    color_str = ''.join(
        _color_char(c) for c in range(num_colors)
        if knowledge.color_plausible(c)
    )
    if not color_str:
      color_str = 'X'

  # Rank knowledge.
  if knowledge.rank() is not None:
    rank_str = str(knowledge.rank() + 1)
  else:
    rank_str = ''.join(
        str(r + 1) for r in range(num_ranks)
        if knowledge.rank_plausible(r)
    )
    if not rank_str:
      rank_str = 'X'

  return f'{color_str}{rank_str}'
