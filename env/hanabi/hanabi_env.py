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
import logging

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

  # In-memory cache for serialize/deserialize round-trips within this
  # process.  Hanabi's stochastic card deals make action-replay
  # deserialization non-deterministic, so we clone instead.
  _serialize_cache: dict[str, 'HanabiState'] = {}

  def __init__(self, game: HanabiGame, hle_state: pyhanabi.HanabiState) -> None:
    self._game = game
    self._hle_state = hle_state
    self._action_history: list[int] = []

  def _advance_past_chance(self) -> None:
    """Advance past any chance nodes (card dealing) in the HLE state.

    In Hanabi, after a Play or Discard action a replacement card is
    dealt from the deck.  The HLE models this as a chance node where
    ``cur_player() == CHANCE_PLAYER_ID``.  This helper resolves all
    pending chance events so the state always lands on a regular
    player node (or terminal).
    """
    while (
        not self._hle_state.is_terminal()
        and self._hle_state.cur_player() == pyhanabi.CHANCE_PLAYER_ID
    ):
      if hasattr(self._hle_state, 'deal_random_card'):
        self._hle_state.deal_random_card()
      elif hasattr(self._hle_state, 'apply_random_chance'):
        self._hle_state.apply_random_chance()
      else:
        break

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

  def score(self) -> int:
    """Current firework score (0-25 for standard game)."""
    return self._hle_state.score()

  def life_tokens(self) -> int:
    """Remaining life tokens (0-3 for standard game)."""
    return self._hle_state.life_tokens()

  def information_tokens(self) -> int:
    """Remaining information tokens (0-8 for standard game)."""
    return self._hle_state.information_tokens()

  def deck_size(self) -> int:
    """Cards remaining in the draw deck."""
    return self._hle_state.deck_size()

  def state_value(self) -> float:
    """Heuristic evaluation of the current game state.

    Returns a value in [0, 25] estimating the expected final score.
    Used by the 'rollout' reward simulation mode to evaluate
    non-terminal states after truncated random playouts.

    The heuristic combines:
      - Current score (fireworks already completed)
      - Remaining potential discounted by:
        - Life tokens (fewer lives = more fragile)
        - Information tokens (fewer tokens = harder to coordinate)
        - Deck exhaustion (fewer cards = less recovery opportunity)
    """
    if self.is_terminal():
      return float(self.score())

    s = self.score()
    lives = self.life_tokens()
    max_lives = self._game._params.get('max_life_tokens', 3)
    max_info = self._game._params.get('max_information_tokens', 8)
    num_colors = self._game._params.get('colors', 5)
    num_ranks = self._game._params.get('ranks', 5)
    max_score = num_colors * num_ranks  # 25 for standard

    if lives == 0:
      return float(s)

    remaining = max_score - s
    # Health factor: 0 lives = 0, full lives = 1.0
    health = lives / max(max_lives, 1)
    # Info factor: diminishing returns, 4+ tokens is fine
    info = min(self.information_tokens() / max(max_info * 0.5, 1), 1.0)
    # Deck factor: more deck = more room to recover from mistakes
    deck = self.deck_size()
    deck_factor = min(deck / 10.0, 1.0)  # 10+ cards remaining = 1.0

    # Weighted combination: current score + discounted potential
    estimated_remaining = remaining * health * 0.4 * (0.5 + 0.3 * info + 0.2 * deck_factor)
    return float(s) + estimated_remaining

  # ── Actions ────────────────────────────────────────────────────────────

  def legal_actions(self, player: int | None = None) -> list[int]:
    """Return list of legal action UIDs for the current player."""
    if player is not None and player != self.current_player():
      return []
    moves = self._hle_state.legal_moves()
    return [self._game._hle_game.get_move_uid(m) for m in moves]

  def action_to_string(self, player: int, action_id: int) -> str:
    """Convert an action UID to a human-readable string.

    Matches the OpenSpiel format:
      "(Play 0)", "(Discard 2)", "(Reveal player +1 color R)",
      "(Reveal player +1 rank 3)"
    """
    move = self._game._hle_game.get_move(action_id)
    return _move_to_string(move, player, self._game.num_players())

  def apply_action(self, action_id: int) -> None:
    """Apply an action and advance the game state.

    After the player's move, any chance events (card dealing after
    play/discard) are resolved automatically so the state always
    lands on a player node or terminal.
    """
    move = self._game._hle_game.get_move(action_id)
    self._hle_state.apply_move(move)
    self._action_history.append(action_id)
    # Advance past chance nodes (card dealing after play/discard).
    self._advance_past_chance()

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
        self._hle_state._state, self._game._hle_game._game, player
    )
    return _format_observation(
        obs, player, self._game._hle_game, self._hle_state
    )

  # ── Serialization ──────────────────────────────────────────────────────

  def serialize(self) -> str:
    """Serialize the state — stores a clone in an in-memory cache."""
    import uuid as _uuid  # pylint: disable=g-import-not-at-top
    state_id = str(_uuid.uuid4())
    HanabiState._serialize_cache[state_id] = self.clone()
    return json.dumps({'state_id': state_id})

  @classmethod
  def deserialize(cls, game: HanabiGame, data_str: str) -> HanabiState:
    """Restore a state from its serialized form (cache lookup)."""
    data = json.loads(data_str)
    state_id = data.get('state_id')
    if state_id and state_id in cls._serialize_cache:
      return cls._serialize_cache[state_id].clone()
    logging.warning('HanabiState cache miss for id=%s', state_id)
    return game.new_initial_state()

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
    # Store constructor-compatible keys for serialization round-trip.
    self._params = {
        'players': players,
        'colors': colors,
        'ranks': ranks,
        'hand_size': hand_size,
        'max_information_tokens': max_information_tokens,
        'max_life_tokens': max_life_tokens,
        'random_start_player': random_start_player,
    }
    # HLE expects the key 'rank' (not 'ranks').
    hle_params = dict(self._params)
    hle_params['rank'] = hle_params.pop('ranks')
    self._hle_game = pyhanabi.HanabiGame(hle_params)

  def num_players(self) -> int:
    return self._params['players']

  def num_distinct_actions(self) -> int:
    """Return the total number of distinct action UIDs."""
    return self._hle_game.max_moves()

  def get_type(self):
    """Return a game-type descriptor matching OpenSpiel's GameType API.

    Only ``short_name`` is used by ``LLMAgent`` for prompt construction.
    """

    class _GameType:  # pylint: disable=invalid-name
      short_name = 'hanabi'

    return _GameType()

  def new_initial_state(self) -> HanabiState:
    """Create a new game state (deals cards randomly)."""
    hle_state = self._hle_game.new_initial_state()
    state = HanabiState(self, hle_state)
    # Advance past initial chance nodes (dealing starting hands).
    state._advance_past_chance()
    return state

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

  @property
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
      actions: A list with a single action ID (matches OpenSpiel's multi-agent
        step convention where only the current player's action slot matters).

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


# Module-level cache for in-memory state cloning.
# Hanabi is stochastic (random card deals between player actions), so
# action-replay deserialization doesn't work — replaying the same player
# actions produces illegal moves because different cards were dealt.
# Instead we clone states in memory and look them up by UUID.
_state_cache: dict[str, tuple[HanabiGame, HanabiState]] = {}


def clear_state_cache() -> None:
  """Clear all state caches (call at the start of each GRPO pass)."""
  _state_cache.clear()
  HanabiState._serialize_cache.clear()


def serialize_game_and_state(game: HanabiGame, state: HanabiState) -> str:
  """Serialize a game+state pair for later restoration.

  Stores a clone of the state in an in-memory cache and returns a
  JSON string containing the cache key.  This avoids the stochastic
  replay problem in Hanabi where random card deals between player
  actions make action-replay deserialization non-deterministic.
  """
  import uuid  # pylint: disable=g-import-not-at-top
  state_id = str(uuid.uuid4())
  _state_cache[state_id] = (game, state.clone())
  return json.dumps({
      'adapter': 'hanabi_env',
      'state_id': state_id,
      'params': game._params,
  })


def deserialize_game_and_state(data_str: str) -> tuple[HanabiGame, HanabiState]:
  """Restore a game+state pair from the in-memory cache.

  Returns a fresh clone each time so the caller can mutate freely.
  """
  data = json.loads(data_str)
  state_id = data.get('state_id')
  if state_id and state_id in _state_cache:
    game, cached_state = _state_cache[state_id]
    return game, cached_state.clone()
  # Fallback: create a fresh initial state (loses mid-game position,
  # but avoids a crash).
  logging.warning(
      'Hanabi state cache miss for id=%s — returning fresh initial state.',
      state_id,
  )
  game = HanabiGame(**data['params'])
  return game, game.new_initial_state()


# ═══════════════════════════════════════════════════════════════════════════
# Internal formatting helpers
# ═══════════════════════════════════════════════════════════════════════════


def _move_to_string(
    move: pyhanabi.HanabiMove, player: int, num_players: int
) -> str:
  """Convert an HLE move to the OpenSpiel action string format."""
  move_type = move.type()

  # HLE defines move types as integers via HanabiMoveType IntEnum:
  #   PLAY=1, DISCARD=2, REVEAL_COLOR=3, REVEAL_RANK=4
  if move_type == 1:  # PLAY
    return f'(Play {move.card_index()})'
  elif move_type == 2:  # DISCARD
    return f'(Discard {move.card_index()})'
  elif move_type == 3:  # REVEAL_COLOR
    target = move.target_offset()
    color = _color_char(move.color())
    return f'(Reveal player +{target} color {color})'
  elif move_type == 4:  # REVEAL_RANK
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

  Note: HLE's ``color()``/``rank()`` return -1 when unknown (not None).
  """
  # Color knowledge.
  color_val = knowledge.color()
  if color_val is not None and color_val >= 0:
    color_str = _color_char(color_val)
  else:
    color_str = ''.join(
        _color_char(c)
        for c in range(num_colors)
        if knowledge.color_plausible(c)
    )
    if not color_str:
      color_str = 'X'

  # Rank knowledge.
  rank_val = knowledge.rank()
  if rank_val is not None and rank_val >= 0:
    rank_str = str(rank_val + 1)
  else:
    rank_str = ''.join(
        str(r + 1) for r in range(num_ranks) if knowledge.rank_plausible(r)
    )
    if not rank_str:
      rank_str = 'X'

  return f'{color_str}{rank_str}'
