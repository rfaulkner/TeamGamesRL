# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rule-based Hanabi players for baselines and curriculum training.

This module provides heuristic players that encode Hanabi domain knowledge.
They serve two purposes:

1. **Training partners**: Agents trained via RL can be partnered with
   heuristic players of varying strength to form a curriculum.
2. **Evaluation baselines**: Comparing a learned agent's score against
   heuristic partners gives a quick sanity-check beyond raw reward.

Classes:
  HeuristicPlayer: Abstract base for all rule-based players.
  RandomPlayer: Uniformly random legal-action selection (Level 0).
  SafePlayPlayer: Prioritises known-safe plays, then hints, then
      discards (Level 1).
"""

from __future__ import annotations

import abc
import re
from typing import List, Optional, Tuple

from env.hanabi.hanabi_env import CARD_KNOWLEDGE_RE as _CARD_KNOWLEDGE_RE
from env.hanabi.hanabi_env import COLOR_CHARS as _COLOR_LETTERS
from env.hanabi.hanabi_env import FIREWORKS_RE as _FIREWORKS_RE
from env.hanabi.hanabi_env import INFO_TOKENS_RE as _INFO_TOKENS_RE
from env.hanabi.hanabi_env import LIFE_TOKENS_RE as _LIFE_TOKENS_RE
from env.hanabi.hanabi_env import parse_fireworks
import numpy as np

# Hanabi constants.
_MAX_INFO_TOKENS = 8


def create_heuristic_player(
    level: int = 1, seed: Optional[int] = None
) -> HeuristicPlayer:
  """Creates a heuristic player for the requested skill level.

  Args:
    level: 0 for RandomPlayer, 1 for SafePlayPlayer.
    seed: Optional RNG seed.

  Returns:
    A HeuristicPlayer instance.
  """
  if level == 0:
    return RandomPlayer(seed=seed)
  elif level == 1:
    return SafePlayPlayer(seed=seed)
  raise ValueError(f'Unknown heuristic level: {level}')


class HeuristicPlayer(abc.ABC):
  """Abstract base class for rule-based Hanabi players.

  Can also be instantiated directly via ``HeuristicPlayer(level=N)``
  which acts as a factory returning the corresponding subclass.

  Subclasses must implement ``select_action`` and the ``name`` property.
  """

  def __new__(cls, *args, **kwargs):
    if cls is HeuristicPlayer:
      level = kwargs.get('level', args[0] if args else 1)
      seed = kwargs.get('seed', args[1] if len(args) > 1 else None)
      return create_heuristic_player(level=level, seed=seed)
    return super().__new__(cls)

  @abc.abstractmethod
  def select_action(
      self,
      state: 'pyspiel.State',
      player_id: int,
      game: Optional['pyspiel.Game'] = None,
  ) -> int:
    """Chooses an action for the given player.

    Args:
      state: The current OpenSpiel game state.
      player_id: Index of the acting player.
      game: Optional OpenSpiel game object (used for action metadata).

    Returns:
      A legal action integer.
    """

  @property
  @abc.abstractmethod
  def name(self) -> str:
    """Short human-readable identifier for this player type."""


class RandomPlayer(HeuristicPlayer):
  """Level-0 baseline that picks a uniformly random legal action.

  Attributes:
    _rng: NumPy random generator used for reproducibility.
  """

  def __init__(self, seed: Optional[int] = None, **kwargs) -> None:
    """Initialises a RandomPlayer.

    Args:
      seed: Optional RNG seed for reproducible action selection.
      **kwargs: Ignored keyword arguments for factory compatibility.
    """
    del kwargs
    self._rng = np.random.RandomState(seed)

  def select_action(
      self,
      state: 'pyspiel.State',
      player_id: int,
      game: Optional['pyspiel.Game'] = None,
  ) -> int:
    """Selects a uniformly random legal action.

    Args:
      state: The current OpenSpiel game state.
      player_id: Index of the acting player (unused; legality comes
          from *state*).
      game: The OpenSpiel game object (unused).

    Returns:
      A randomly chosen legal action integer.
    """
    legal_actions = state.legal_actions(player_id)
    return int(self._rng.choice(legal_actions))

  @property
  def name(self) -> str:
    return 'random'


# Where does a freshly drawn card land in a hand?
#
# Verified directly against HLE in scratch/probe_hand_order.py: hands are
# ordered oldest -> newest and a replacement card is APPENDED to the end, so
# the OLDEST card sits at index 0.  The original implementation assumed the
# opposite ("new cards are inserted at position 0") and therefore discarded
# the NEWEST card whenever it meant to discard the oldest.  A 300-deal paired
# experiment put the cost of that inversion at +1.30 +/- 0.32 points out of 25
# (4.16 corrected vs 2.86 as-written).
#
# Set this True only to reproduce runs made before the fix.
_NEW_CARDS_AT_INDEX_ZERO = False


class SafePlayPlayer(HeuristicPlayer):
  """Level-1 heuristic that prioritises safe plays over random hints.

  Decision priority (highest → lowest):
    1. Play a card that is **known** to be playable on the current
       fireworks.
    2. If information tokens < 8 and there is an unhinted card,
       discard the oldest unhinted card.
    3. If information tokens > 0, give a random legal hint (Reveal
       action).
    4. Otherwise discard the oldest card in hand.

  A card is considered *known playable* when exactly one colour and
  one rank remain in its knowledge, and that (colour, rank) pair is
  the next needed card for the corresponding firework.
  """

  def __init__(
      self,
      seed: Optional[int] = None,
      new_cards_at_index_zero: bool = _NEW_CARDS_AT_INDEX_ZERO,
      **kwargs,
  ) -> None:
    """Initialises a SafePlayPlayer.

    Args:
      seed: Optional RNG seed for tie-breaking and hint selection.
      new_cards_at_index_zero: Whether a freshly drawn card is inserted at
        hand position 0.  False matches HLE, where replacements are appended
        and index 0 is the oldest card.  Pass True only to reproduce runs
        made before this was corrected.
      **kwargs: Ignored keyword arguments for factory compatibility.
    """
    del kwargs
    self._rng = np.random.RandomState(seed)
    self._new_cards_at_index_zero = new_cards_at_index_zero

  # ---------------------------------------------------------------------------
  # Public interface
  # ---------------------------------------------------------------------------

  def select_action(
      self,
      state: 'pyspiel.State',
      player_id: int,
      game: Optional['pyspiel.Game'] = None,
  ) -> int:
    """Selects an action using the safe-play heuristic.

    Args:
      state: The current OpenSpiel game state.
      player_id: Index of the acting player.
      game: The OpenSpiel game object (used for action string
          conversion).

    Returns:
      A legal action integer chosen by the heuristic priority.
    """
    legal_actions = state.legal_actions(player_id)
    obs_string = state.observation_string(player_id)

    fireworks = self._parse_fireworks(obs_string)
    info_tokens = self._parse_info_tokens(obs_string)
    card_knowledge = self._parse_card_knowledge(obs_string)

    # Build convenience maps from action → type / position.
    play_actions, discard_actions, hint_actions = self._classify_actions(
        state, player_id, legal_actions
    )

    # Priority 1: Play a known-playable card.
    playable_action = self._find_known_playable(
        card_knowledge, fireworks, play_actions
    )
    if playable_action is not None:
      return playable_action

    # Priority 2: Discard the oldest unhinted card (if info tokens
    # are not full).
    if info_tokens < _MAX_INFO_TOKENS and discard_actions:
      oldest_unhinted = self._oldest_unhinted_card(
          card_knowledge, discard_actions, self._new_cards_at_index_zero
      )
      if oldest_unhinted is not None:
        return oldest_unhinted

    # Priority 3: Give a random legal hint if we have info tokens.
    if info_tokens > 0 and hint_actions:
      return int(self._rng.choice(hint_actions))

    # Priority 4: Discard the oldest card.  Under HLE ordering that is the
    # LOWEST position index, not the highest.
    if discard_actions:
      pick = max if self._new_cards_at_index_zero else min
      return pick(discard_actions, key=lambda a: discard_actions[a])

    # Fallback (should not happen in a valid game state).
    return int(self._rng.choice(legal_actions))

  @property
  def name(self) -> str:
    return 'safe_play'

  # ---------------------------------------------------------------------------
  # Observation parsing helpers
  # ---------------------------------------------------------------------------

  @staticmethod
  def _parse_fireworks(obs_string: str) -> dict[str, int]:
    return parse_fireworks(obs_string)

  @staticmethod
  def _parse_info_tokens(obs_string: str) -> int:
    """Extracts the current information-token count.

    Args:
      obs_string: The raw observation string from OpenSpiel.

    Returns:
      The number of information tokens remaining.
    """
    match = _INFO_TOKENS_RE.search(obs_string)
    return int(match.group(1)) if match else _MAX_INFO_TOKENS

  @staticmethod
  def _parse_card_knowledge(
      obs_string: str,
  ) -> List[Tuple[str, str]]:
    """Parses the acting player's card knowledge from the observation.

    Each card the player holds appears as ``XX || <colors>|<ranks>``
    in the observation string (``XX`` because the player cannot see
    their own cards).

    Args:
      obs_string: The raw observation string from OpenSpiel.

    Returns:
      A list of ``(possible_colors, possible_ranks)`` tuples, one per
      card in hand order (index 0 = newest, highest index = oldest).
      ``possible_colors`` is a string of remaining colour letters
      (e.g. ``'R'`` if known red).  ``possible_ranks`` is a string
      of remaining rank digits (e.g. ``'3'`` if known rank 3).
    """
    return _CARD_KNOWLEDGE_RE.findall(obs_string)

  # ---------------------------------------------------------------------------
  # Action classification helpers
  # ---------------------------------------------------------------------------

  @staticmethod
  def _classify_actions(
      state: 'pyspiel.State',
      player_id: int,
      legal_actions: List[int],
  ) -> Tuple[dict[int, int], dict[int, int], List[int]]:
    """Splits legal actions into play, discard, and hint buckets.

    Args:
      state: The current game state (used for
          ``action_to_string``).
      player_id: The acting player's index.
      legal_actions: List of legal action integers.

    Returns:
      A 3-tuple of:
        - play_actions: ``{action_int: card_position}`` for Play
          actions.
        - discard_actions: ``{action_int: card_position}`` for
          Discard actions.
        - hint_actions: list of action ints for Reveal actions.
    """
    play_actions: dict[int, int] = {}
    discard_actions: dict[int, int] = {}
    hint_actions: List[int] = []

    play_re = re.compile(r'\(Play (\d+)\)')
    discard_re = re.compile(r'\(Discard (\d+)\)')

    for action in legal_actions:
      action_str = state.action_to_string(player_id, action)
      play_match = play_re.search(action_str)
      if play_match:
        play_actions[action] = int(play_match.group(1))
        continue
      discard_match = discard_re.search(action_str)
      if discard_match:
        discard_actions[action] = int(discard_match.group(1))
        continue
      if 'Reveal' in action_str:
        hint_actions.append(action)

    return play_actions, discard_actions, hint_actions

  # ---------------------------------------------------------------------------
  # Decision logic helpers
  # ---------------------------------------------------------------------------

  @staticmethod
  def _find_known_playable(
      card_knowledge: List[Tuple[str, str]],
      fireworks: dict[str, int],
      play_actions: dict[int, int],
  ) -> Optional[int]:
    """Returns a Play action for a card known to be playable, or None.

    A card is *known playable* when its knowledge narrows to exactly
    one colour and one rank, and that rank equals
    ``fireworks[colour] + 1``.

    Args:
      card_knowledge: Per-card ``(colors, ranks)`` knowledge tuples.
      fireworks: Current firework heights per colour.
      play_actions: ``{action_int: card_position}`` map.

    Returns:
      The action integer for the first known-playable card found
      (scanning from lowest position index), or ``None``.
    """
    for action, position in sorted(play_actions.items(), key=lambda x: x[1]):
      if position >= len(card_knowledge):
        continue
      colors, ranks = card_knowledge[position]
      if len(colors) == 1 and len(ranks) == 1:
        needed_rank = fireworks.get(colors, 0) + 1
        if int(ranks) == needed_rank:
          return action
    return None

  @staticmethod
  def _oldest_unhinted_card(
      card_knowledge: List[Tuple[str, str]],
      discard_actions: dict[int, int],
      new_cards_at_index_zero: bool = _NEW_CARDS_AT_INDEX_ZERO,
  ) -> Optional[int]:
    """Returns a Discard action for the oldest card with no hints.

    A card is considered *unhinted* when its knowledge still contains
    all five colours **and** all five ranks (i.e., no Reveal action
    has touched it).

    Which index is "oldest" depends on where the environment inserts a
    freshly drawn card.  HLE appends replacements, so the oldest card is at
    the LOWEST index; when ``new_cards_at_index_zero`` is True the oldest is
    instead at the highest index.

    Args:
      card_knowledge: Per-card ``(colors, ranks)`` knowledge tuples.
      discard_actions: ``{action_int: card_position}`` map.
      new_cards_at_index_zero: Whether new cards are inserted at position 0.

    Returns:
      The action integer for discarding the oldest unhinted card, or
      ``None`` if every card has received at least one hint.
    """
    unhinted: List[Tuple[int, int]] = []  # (action, position)
    for action, position in discard_actions.items():
      if position >= len(card_knowledge):
        continue
      colors, ranks = card_knowledge[position]
      if len(colors) == len(_COLOR_LETTERS) and len(ranks) == 5:
        unhinted.append((action, position))
    if not unhinted:
      return None
    pick = max if new_cards_at_index_zero else min
    return pick(unhinted, key=lambda x: x[1])[0]
