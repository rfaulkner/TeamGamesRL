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

"""Belief-consistent determinization of Hanabi states.

Why this exists
---------------
``_heuristic_rollout_score`` evaluates a candidate action by cloning the *true*
state and rolling it forward.  The true state contains the acting player's own
cards, which that player cannot see.  The resulting reward therefore scores
moves that are good *given hidden information the policy does not have*, which
trains the policy to gamble.  Determinization fixes this: instead of one
rollout from the true world, average rollouts over many worlds that are
consistent with everything the player can actually observe.

How it works
------------
HLE will not let us write a hand directly -- there is no ``StateSetHandCard``
and no way to construct a DEAL move.  But a DEAL move can be *harvested* from
``move_history()`` and re-applied at a chance node, so a game can be replayed
move-for-move with chosen cards substituted for the acting player's own hand.

Correctness
-----------
``move_is_legal`` prevents crashes but does NOT guarantee belief consistency: a
past "reveal red" stays legal as long as *some* card is red, so a naive
substitution can silently mark different slots.  A candidate world is therefore
accepted only when the acting player's card knowledge *and* every public field
are identical to the real state.

Danger
------
Several HLE misuses abort the process via C++ assertions rather than raising,
so they cannot be caught:

  * ``legal_moves()`` at a chance node,
  * ``apply_move()`` with an illegal move (always gate on ``move_is_legal``),
  * letting a ``HanabiGame`` be garbage-collected while a state derived from it
    is still alive (``__del__`` calls ``DeleteGame``),
  * holding a ``HanabiCardKnowledge`` (or any other view) after the
    ``HanabiObservation`` that owns it has been collected -- so never write
    ``obs_expr().card_knowledge()``, always bind the observation first.

Every call in this module is written to respect those constraints.
"""

from __future__ import annotations

import collections
import random
from typing import Any, Optional

try:
  from hanabi_learning_environment import pyhanabi
  _MT = pyhanabi.HanabiMoveType
except ImportError:
  pyhanabi = None
  _MT = None

# Number of random games used to collect all distinct deal moves.  Six is
# typically enough for a standard 5x5 deck; the loop is bounded anyway.
_MAX_HARVEST_GAMES = 500


class DeterminizationError(RuntimeError):
  """Raised when a state cannot be determinized."""


class Determinizer:
  """Resamples a player's own hand while preserving everything observable.

  One instance is bound to one ``HanabiGame``.  The game object must outlive
  the determinizer and every state it produces.
  """

  def __init__(self, game: Any, harvest_seed: int = 0) -> None:
    """Builds the deal-move table for ``game``.

    Args:
      game: Either a ``env.hanabi.hanabi_env.HanabiGame`` wrapper or a raw
        ``pyhanabi.HanabiGame``.
      harvest_seed: Seed for the random games used to harvest deal moves.
    """
    self._hle_game = getattr(game, '_hle_game', game)
    self._wrapper_game = game if hasattr(game, '_hle_game') else None
    self._num_colors = self._hle_game.num_colors()
    self._num_ranks = self._hle_game.num_ranks()
    self._deal_moves = self._harvest_deal_moves(harvest_seed)

  # ── setup ───────────────────────────────────────────────────────────────

  def _harvest_deal_moves(self, seed: int) -> dict[tuple[int, int],
                                                   pyhanabi.HanabiMove]:
    """Collects one DEAL move object per (color, rank).

    DEAL moves cannot be constructed, only observed, so we play random games
    until every card has been seen dealt at least once.
    """
    wanted = self._num_colors * self._num_ranks
    moves: dict[tuple[int, int], pyhanabi.HanabiMove] = {}
    rng = random.Random(seed)
    for _ in range(_MAX_HARVEST_GAMES):
      if len(moves) >= wanted:
        break
      state = self._hle_game.new_initial_state()
      while not state.is_terminal():
        if state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
          state.deal_random_card()
          continue
        state.apply_move(rng.choice(state.legal_moves()))
      for item in state.move_history():
        move = item.move()
        if move.type() == _MT.DEAL:
          moves.setdefault((move.color(), move.rank()), move)
    if len(moves) < wanted:
      raise DeterminizationError(
          f'harvested only {len(moves)}/{wanted} deal moves')
    return moves

  # ── history handling ────────────────────────────────────────────────────

  @staticmethod
  def _history_tuples(hle_state) -> list[tuple]:
    """Flattens a move history into re-appliable plain tuples."""
    out = []
    for item in hle_state.move_history():
      move = item.move()
      kind = move.type()
      if kind == _MT.DEAL:
        out.append(('deal', move.color(), move.rank(), item.deal_to_player()))
      elif kind == _MT.PLAY:
        out.append(('play', move.card_index()))
      elif kind == _MT.DISCARD:
        out.append(('discard', move.card_index()))
      elif kind == _MT.REVEAL_COLOR:
        out.append(('rc', move.target_offset(), move.color()))
      elif kind == _MT.REVEAL_RANK:
        out.append(('rr', move.target_offset(), move.rank()))
    return out

  def _rebuild(self, item: tuple) -> pyhanabi.HanabiMove:
    kind = item[0]
    if kind == 'deal':
      return self._deal_moves[(item[1], item[2])]
    if kind == 'play':
      return pyhanabi.HanabiMove.get_play_move(item[1])
    if kind == 'discard':
      return pyhanabi.HanabiMove.get_discard_move(item[1])
    if kind == 'rc':
      return pyhanabi.HanabiMove.get_reveal_color_move(item[1], item[2])
    if kind == 'rr':
      return pyhanabi.HanabiMove.get_reveal_rank_move(item[1], item[2])
    raise ValueError(f'unknown history item {item!r}')

  def _replay(self, hist: list[tuple]):
    """Replays a history into a fresh state, or returns None if impossible."""
    state = self._hle_game.new_initial_state()
    for item in hist:
      move = self._rebuild(item)
      if not state.move_is_legal(move):
        return None
      state.apply_move(move)
    return state

  def _hand_slot_history_indices(self, hist: list[tuple],
                                 player: int) -> list[int]:
    """Maps each current hand position to the history deal that produced it.

    HLE hands run oldest -> newest and replacements are appended, so a deal
    appends and a play/discard removes at the acted-on index.
    """
    state = self._hle_game.new_initial_state()
    slots: list[int] = []
    for i, item in enumerate(hist):
      actor = state.cur_player()
      if item[0] == 'deal':
        if item[3] == player:
          slots.append(i)
      elif item[0] in ('play', 'discard') and actor == player:
        del slots[item[1]]
      state.apply_move(self._rebuild(item))
    return slots

  # ── observation signatures ──────────────────────────────────────────────

  def _observation(self, hle_state, player: int):
    return pyhanabi.HanabiObservation(
        hle_state._state, self._hle_game._game, player)  # pylint: disable=protected-access

  def _knowledge_signature(self, hle_state, player: int) -> tuple:
    """The acting player's own card knowledge, as a comparable tuple."""
    obs = self._observation(hle_state, player)
    return tuple(
        (k.color(), k.rank(),
         tuple(c for c in range(self._num_colors) if k.color_plausible(c)),
         tuple(r for r in range(self._num_ranks) if k.rank_plausible(r)))
        for k in obs.card_knowledge()[0]
    )

  def _public_signature(self, hle_state, player: int) -> tuple:
    """Everything ``player`` can legitimately observe."""
    obs = self._observation(hle_state, player)
    return (
        tuple(obs.fireworks()),
        tuple(sorted((c.color(), c.rank()) for c in obs.discard_pile())),
        obs.life_tokens(),
        obs.information_tokens(),
        obs.deck_size(),
        tuple(tuple((c.color(), c.rank()) for c in hand)
              for hand in obs.observed_hands()[1:]),
        hle_state.cur_player(),
    )

  def unseen_pool(self, hle_state, player: int) -> collections.Counter:
    """Cards ``player`` cannot see: their own hand plus the remaining deck."""
    obs = self._observation(hle_state, player)
    pool = collections.Counter(
        (c, r)
        for c in range(self._num_colors)
        for r in range(self._num_ranks)
        for _ in range(self._hle_game.num_cards(c, r))
    )
    for card in obs.discard_pile():
      pool[(card.color(), card.rank())] -= 1
    for color, height in enumerate(obs.fireworks()):
      for rank in range(height):  # ranks 0..height-1 are on the stack
        pool[(color, rank)] -= 1
    for hand in obs.observed_hands()[1:]:
      for card in hand:
        pool[(card.color(), card.rank())] -= 1
    return +pool  # drops zero and negative entries

  # ── the main entry point ────────────────────────────────────────────────

  def determinize(
      self,
      state: Any,
      player: int,
      rng: random.Random,
      max_attempts: int = 100,
  ):
    """Returns a state identical to ``state`` except for ``player``'s hand.

    Args:
      state: A wrapper ``HanabiState`` or a raw ``pyhanabi.HanabiState``.
      player: The player whose hand is resampled.
      rng: Source of randomness.
      max_attempts: Rejection-sampling budget.

    Returns:
      A new state of the same kind as ``state``, or None if no consistent
      world was found within the budget.
    """
    hle_state = getattr(state, '_hle_state', state)
    hist = self._history_tuples(hle_state)
    target_knowledge = self._knowledge_signature(hle_state, player)
    target_public = self._public_signature(hle_state, player)
    # The observation MUST stay referenced for as long as `knowledge` is used:
    # HanabiCardKnowledge points into memory owned by the observation, and
    # HanabiObservation.__del__ frees it.  Binding the observation to a
    # temporary here caused a use-after-free segfault inside color_plausible.
    observation = self._observation(hle_state, player)
    knowledge = observation.card_knowledge()[0]
    hand_size = len(knowledge)

    slots = self._hand_slot_history_indices(hist, player)
    if len(slots) != hand_size:
      raise DeterminizationError(
          f'tracked {len(slots)} hand slots but hand holds {hand_size}')

    pool = self.unseen_pool(hle_state, player)

    # Fill the most constrained slot first so we rarely paint ourselves into a
    # corner; weight by multiplicity so the sample respects card counts.
    def plausible_count(i: int) -> int:
      return sum(1
                 for c in range(self._num_colors)
                 for r in range(self._num_ranks)
                 if knowledge[i].color_plausible(c)
                 and knowledge[i].rank_plausible(r))

    order = sorted(range(hand_size), key=plausible_count)

    for _ in range(max_attempts):
      available = collections.Counter(pool)
      picked: dict[int, tuple[int, int]] = {}
      dead_end = False
      for i in order:
        candidates = [
            cr for cr, n in available.items()
            if n > 0
            and knowledge[i].color_plausible(cr[0])
            and knowledge[i].rank_plausible(cr[1])
        ]
        if not candidates:
          dead_end = True
          break
        choice = rng.choices(candidates,
                             weights=[available[cr] for cr in candidates],
                             k=1)[0]
        picked[i] = choice
        available[choice] -= 1
      if dead_end:
        continue

      alt = list(hist)
      for position, hist_index in enumerate(slots):
        color, rank = picked[position]
        alt[hist_index] = ('deal', color, rank, player)

      candidate = self._replay(alt)
      if candidate is None:
        continue
      if self._knowledge_signature(candidate, player) != target_knowledge:
        continue
      if self._public_signature(candidate, player) != target_public:
        continue
      return self._wrap(candidate, state)

    return None

  def _wrap(self, hle_state, like: Any):
    """Returns ``hle_state`` in the same form as ``like``."""
    if hasattr(like, '_hle_state'):
      wrapper = type(like)(like._game, hle_state)  # pylint: disable=protected-access
      wrapper._action_history = list(like._action_history)  # pylint: disable=protected-access
      return wrapper
    return hle_state
