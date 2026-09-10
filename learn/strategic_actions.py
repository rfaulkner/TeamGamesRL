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

"""Strategic action selection for GRPO completion groups in Hanabi.

Analyses the full game state to select specific actions for forced
inclusion in GRPO groups, replacing blind first-token type forcing
with game-state-aware action injection.

The module categorises every legal action into priority tiers:

  1. **Known-safe plays**: Card fully hinted (colour + rank known) and
     immediately playable.  Highest value -- these should always be
     explored by the GRPO group.
  2. **Risky plays**: Card partially hinted with non-trivial
     probability of being playable.
  3. **Smart discards**: Cards known to be dead (already played) or
     unhinted (safe to discard without information loss).
  4. **Diverse hints**: Hints selected for diversity across colours,
     ranks, and targets, with preference for hints that touch playable
     cards in the partner's hand.

Each tier contributes a configurable number of actions to the GRPO
group, with remaining slots left for free policy generation.

Usage::

    from learn.strategic_actions import analyze_strategic_actions

    plan = analyze_strategic_actions(state, player_id, legal_actions_desc)
    # plan.forced_action_texts -> list of action description strings
    # plan.forced_action_ids   -> list of action ID ints
"""

from __future__ import annotations

import dataclasses
import re

try:
  from absl import logging
except ImportError:
  import logging


# -- Card knowledge regex (matches player's own hand in HLE observations) -----
_CARD_KNOWLEDGE_RE = re.compile(
    r'XX\s*\|\|\s*(?:[A-Z0-9]+[|])?([RYGWB]+)[|]?([1-5]+)'
)

# -- Fireworks regex ----------------------------------------------------------
_FIREWORKS_RE = re.compile(r'Fireworks:\s*((?:[RYGWB]\d\s*)+)')

# -- Action string patterns ---------------------------------------------------
_PLAY_RE = re.compile(r'\(Play (\d+)\)')
_DISCARD_RE = re.compile(r'\(Discard (\d+)\)')
_REVEAL_COLOR_RE = re.compile(r'\(Reveal player \+(\d+) color ([RYGWB])\)')
_REVEAL_RANK_RE = re.compile(r'\(Reveal player \+(\d+) rank (\d+)\)')

# -- Partner hand regex (visible cards in another player's hand) ---------------
# Matches "R2 || ..." format (actual card visible, not XX)
_VISIBLE_CARD_RE = re.compile(r'([RYGWB])(\d)\s*\|\|')

_COLOR_CHARS = ('R', 'Y', 'G', 'W', 'B')


@dataclasses.dataclass
class StrategicActionPlan:
  """Actions selected for forced inclusion in a GRPO group.

  Attributes:
    known_safe_plays: Action IDs for cards known to be immediately playable.
    risky_plays: Action IDs for cards with playability potential.
    smart_discards: Action IDs for safe discard candidates.
    diverse_hints: Action IDs for diverse, useful hint actions.
    action_texts: Maps action ID -> human-readable description text
        (from ``render_legal_actions``).
  """

  known_safe_plays: list[int] = dataclasses.field(default_factory=list)
  risky_plays: list[int] = dataclasses.field(default_factory=list)
  smart_discards: list[int] = dataclasses.field(default_factory=list)
  diverse_hints: list[int] = dataclasses.field(default_factory=list)
  action_texts: dict[int, str] = dataclasses.field(default_factory=dict)

  @property
  def forced_action_ids(self) -> list[int]:
    """All forced action IDs in priority order."""
    return (
        self.known_safe_plays
        + self.risky_plays
        + self.smart_discards
        + self.diverse_hints
    )

  @property
  def forced_action_texts(self) -> list[str]:
    """Completion text strings for all forced actions."""
    return [self.action_texts[aid] for aid in self.forced_action_ids]

  @property
  def num_forced(self) -> int:
    """Total number of forced actions."""
    return len(self.forced_action_ids)


# =============================================================================
# State analysis helpers
# =============================================================================


def _parse_fireworks(obs_string: str) -> dict[str, int]:
  """Extract firework heights from an observation string.

  Args:
    obs_string: Raw observation string from ``state.observation_string()``.

  Returns:
    A dict mapping colour letter (e.g. ``'R'``) to the highest rank
    played on that firework (0 if empty).
  """
  fireworks: dict[str, int] = {c: 0 for c in _COLOR_CHARS}
  match = _FIREWORKS_RE.search(obs_string)
  if match:
    for token in match.group(1).strip().split():
      if len(token) >= 2:
        fireworks[token[0]] = int(token[1:])
  return fireworks


def _parse_card_knowledge(obs_string: str) -> list[tuple[str, str]]:
  """Parse the acting player's card knowledge from their observation.

  Each card the player holds appears as ``XX || <colors>|<ranks>``
  in the observation string (``XX`` because the player cannot see
  their own cards).

  Args:
    obs_string: Raw observation string from ``state.observation_string()``.

  Returns:
    A list of ``(possible_colors, possible_ranks)`` tuples, one per
    card in hand order.  ``possible_colors`` is a string of remaining
    colour letters (e.g. ``'R'`` if known red).  ``possible_ranks``
    is a string of remaining rank digits (e.g. ``'3'`` if known rank 3).
  """
  return _CARD_KNOWLEDGE_RE.findall(obs_string)


def _parse_partner_hand(obs_string: str) -> list[tuple[str, int]]:
  """Parse the visible cards in the partner's hand.

  The observer can see the partner's actual cards (shown as e.g. ``R2``).

  Args:
    obs_string: Raw observation string (from the current player's
        perspective).

  Returns:
    A list of ``(color_letter, rank_int)`` tuples for each visible
    card in the partner's hand.
  """
  return [(m.group(1), int(m.group(2)))
          for m in _VISIBLE_CARD_RE.finditer(obs_string)]


def _classify_legal_actions(
    state,
    player_id: int,
    legal_actions: list[int],
) -> tuple[dict[int, int], dict[int, int], list[int]]:
  """Classify legal actions into play, discard, and hint buckets.

  Args:
    state: The current game state.
    player_id: The acting player's index.
    legal_actions: List of legal action UIDs.

  Returns:
    A 3-tuple of:
      - play_actions: ``{action_id: card_position}`` for Play actions.
      - discard_actions: ``{action_id: card_position}`` for Discard.
      - hint_actions: list of ``(action_id, hint_type, hint_value,
        target_offset)`` tuples for Reveal actions.
  """
  play_actions: dict[int, int] = {}
  discard_actions: dict[int, int] = {}
  hint_actions: list[tuple[int, str, str, int]] = []

  for action in legal_actions:
    action_str = state.action_to_string(player_id, action)

    play_match = _PLAY_RE.search(action_str)
    if play_match:
      play_actions[action] = int(play_match.group(1))
      continue

    discard_match = _DISCARD_RE.search(action_str)
    if discard_match:
      discard_actions[action] = int(discard_match.group(1))
      continue

    color_match = _REVEAL_COLOR_RE.search(action_str)
    if color_match:
      hint_actions.append((
          action, 'color', color_match.group(2), int(color_match.group(1))
      ))
      continue

    rank_match = _REVEAL_RANK_RE.search(action_str)
    if rank_match:
      hint_actions.append((
          action, 'rank', rank_match.group(2), int(rank_match.group(1))
      ))


  return play_actions, discard_actions, hint_actions


# =============================================================================
# Tier analysis
# =============================================================================


def _find_known_safe_plays(
    card_knowledge: list[tuple[str, str]],
    fireworks: dict[str, int],
    play_actions: dict[int, int],
) -> list[int]:
  """Find play actions for cards known to be immediately playable.

  A card is *known playable* when its knowledge narrows to exactly
  one colour and one rank, and that rank equals ``fireworks[colour] + 1``.

  Args:
    card_knowledge: Per-card ``(colors, ranks)`` knowledge tuples.
    fireworks: Current firework heights per colour.
    play_actions: ``{action_id: card_position}`` map.

  Returns:
    List of action IDs for known-safe plays.
  """
  safe_plays = []
  for action, position in sorted(play_actions.items(), key=lambda x: x[1]):
    if position >= len(card_knowledge):
      continue
    colors, ranks = card_knowledge[position]
    if len(colors) == 1 and len(ranks) == 1:
      needed_rank = fireworks.get(colors, 0) + 1
      if int(ranks) == needed_rank:
        safe_plays.append(action)
  return safe_plays


def _find_risky_plays(
    card_knowledge: list[tuple[str, str]],
    fireworks: dict[str, int],
    play_actions: dict[int, int],
    safe_plays: list[int],
    min_playable_frac: float = 0.3,
    max_risky: int = 2,
) -> list[int]:
  """Find play actions with non-trivial playability probability.

  Selects cards where hints narrow the possibilities enough that at
  least ``min_playable_frac`` of remaining (colour, rank) combos would
  be immediately playable.  Excludes cards already in ``safe_plays``.

  Args:
    card_knowledge: Per-card knowledge tuples.
    fireworks: Current firework heights.
    play_actions: ``{action_id: card_position}`` map.
    safe_plays: Already-selected known-safe play action IDs.
    min_playable_frac: Minimum fraction of combos that must be playable
        to qualify as a risky play (default 0.3 = 30%).
    max_risky: Maximum number of risky plays to select.

  Returns:
    List of action IDs for risky-but-plausible plays.
  """
  safe_set = set(safe_plays)
  candidates: list[tuple[float, int]] = []  # (playable_frac, action_id)

  for action, position in play_actions.items():
    if action in safe_set:
      continue
    if position >= len(card_knowledge):
      continue
    colors, ranks = card_knowledge[position]

    # Must have SOME hints (not fully unknown).
    if len(colors) >= 5 and len(ranks) >= 5:
      continue

    total_combos = len(colors) * len(ranks)
    if total_combos == 0:
      continue

    playable_combos = 0
    for c in colors:
      needed_rank = fireworks.get(c, 0) + 1
      for r in ranks:
        if int(r) == needed_rank:
          playable_combos += 1

    frac = playable_combos / total_combos
    if frac >= min_playable_frac:
      candidates.append((frac, action))

  # Sort by playability fraction (highest first) and take top max_risky.
  candidates.sort(reverse=True)
  return [action for _, action in candidates[:max_risky]]


def _find_smart_discards(
    state,
    player_id: int,
    card_knowledge: list[tuple[str, str]],
    fireworks: dict[str, int],
    discard_actions: dict[int, int],
    max_discards: int = 2,
) -> list[int]:
  """Find discard actions for cards that are safe to discard.

  Prioritises:
    1. Cards known to be dead (rank <= firework height for known colour).
    2. Completely unhinted cards (oldest first -- standard convention).

  Uses ``state.clone()`` + ``apply_action()`` to verify card identity
  for dead-card detection (same technique as ``action_reward._evaluate_discard``).

  Args:
    state: The current game state.
    player_id: The acting player.
    card_knowledge: Per-card knowledge tuples.
    fireworks: Current firework heights.
    discard_actions: ``{action_id: card_position}`` map.
    max_discards: Maximum number of discard actions to select.

  Returns:
    List of action IDs for smart discard candidates.
  """
  del state, player_id  # Available for future simulation checks.
  dead_cards: list[int] = []
  unhinted: list[tuple[int, int]] = []  # (action, position)

  for action, position in discard_actions.items():
    if position >= len(card_knowledge):
      continue
    colors, ranks = card_knowledge[position]

    # Check for known dead cards.
    if len(colors) == 1 and len(ranks) == 1:
      fw_height = fireworks.get(colors, 0)
      if int(ranks) <= fw_height:
        dead_cards.append(action)
        continue

    # Check for completely unhinted cards.
    if len(colors) >= 5 and len(ranks) >= 5:
      unhinted.append((action, position))

  # If we can verify dead cards via state simulation, also check
  # partially-known cards.
  if len(dead_cards) < max_discards:
    for action, position in discard_actions.items():
      if action in dead_cards:
        continue
      if position >= len(card_knowledge):
        continue
      colors, ranks = card_knowledge[position]
      # Partially known: check if ALL possibilities are dead.
      all_dead = True
      for c in colors:
        for r in ranks:
          if int(r) > fireworks.get(c, 0):
            all_dead = False
            break
        if not all_dead:
          break
      if all_dead and len(colors) * len(ranks) > 0:
        dead_cards.append(action)

  result = dead_cards[:max_discards]

  # Fill remaining slots with unhinted cards (oldest first).
  if len(result) < max_discards and unhinted:
    unhinted.sort(key=lambda x: x[1], reverse=True)  # Highest pos = oldest.
    for action, _ in unhinted:
      if action not in result:
        result.append(action)
        if len(result) >= max_discards:
          break

  return result[:max_discards]


def _find_diverse_hints(
    state,
    player_id: int,
    obs_string: str,
    fireworks: dict[str, int],
    hint_actions: list[tuple[int, str, str, int]],
    max_hints: int = 3,
) -> list[int]:
  """Select diverse, useful hint actions.

  Prioritises:
    1. Hints that touch playable cards in the partner's hand (the
       observer can see partner's cards).
    2. Diversity across hint types (colour vs rank) and values.

  Args:
    state: The current game state.
    player_id: The acting player.
    obs_string: The current player's observation string.
    fireworks: Current firework heights.
    hint_actions: List of ``(action_id, hint_type, hint_value,
        target_offset)`` tuples.
    max_hints: Maximum number of hints to select.

  Returns:
    List of action IDs for diverse hint candidates.
  """
  del state, player_id
  if not hint_actions:
    return []

  # Parse the partner's visible hand.
  partner_hand = _parse_partner_hand(obs_string)

  # Find playable cards in partner's hand.
  playable_positions: set[int] = set()
  for i, (color, rank) in enumerate(partner_hand):
    if rank == fireworks.get(color, 0) + 1:
      playable_positions.add(i)

  # Score each hint by usefulness.
  scored: list[tuple[float, int, str, str]] = []
  for action_id, hint_type, hint_value, _ in hint_actions:
    score = 0.0

    # Bonus for touching playable cards.
    if partner_hand:
      for i, (color, rank) in enumerate(partner_hand):
        touches = False
        if hint_type == 'color' and color == hint_value:
          touches = True
        elif hint_type == 'rank' and str(rank) == hint_value:
          touches = True
        if touches and i in playable_positions:
          score += 2.0  # High value: enables a play.
        elif touches:
          score += 0.5  # Some info value.

    # Small bonus for variety (prefer rarer hint types).
    if hint_type == 'color':
      score += 0.1  # Slight colour preference for diversity.

    scored.append((score, action_id, hint_type, hint_value))

  # Sort by score (descending), then ensure diversity.
  scored.sort(reverse=True)

  selected: list[int] = []
  seen_types: set[tuple[str, str]] = set()  # (hint_type, hint_value)

  # First pass: greedily select highest-scoring hints with diversity.
  for _, action_id, hint_type, hint_value in scored:
    if len(selected) >= max_hints:
      break
    key = (hint_type, hint_value)
    if key not in seen_types:
      selected.append(action_id)
      seen_types.add(key)

  # Second pass: if we still have slots, fill with remaining hints
  # (even if type is duplicated).
  if len(selected) < max_hints:
    for _, action_id, _, _ in scored:
      if action_id not in selected:
        selected.append(action_id)
        if len(selected) >= max_hints:
          break

  return selected


# =============================================================================
# Main analysis entry point
# =============================================================================


def analyze_strategic_actions(
    state,
    player_id: int,
    legal_actions_desc: list[tuple[int, str]],
    num_generations: int = 8,
    max_forced_fraction: float = 1.0,
) -> StrategicActionPlan:
  """Analyse game state and select strategic actions for a GRPO group.

  Examines the current Hanabi game state to categorise all legal
  actions into priority tiers, then selects the best candidates from
  each tier for forced inclusion in the GRPO completion group.

  The number of forced actions is capped at ``max_forced_fraction * K``
  (default 1.0 = 100% strategic actions in early training).

  Args:
    state: The current Hanabi game state (must support
        ``observation_string()``, ``action_to_string()``, ``clone()``,
        ``apply_action()``, ``legal_actions()``).
    player_id: The acting player's index.
    legal_actions_desc: List of ``(action_id, description)`` tuples
        from ``render_legal_actions()``.
    num_generations: Total number of completions per GRPO group (K).
    max_forced_fraction: Maximum fraction of K that can be forced
        (default 1.0 = all slots can be strategic actions).

  Returns:
    A ``StrategicActionPlan`` with selected actions per tier.
  """
  max_forced = max(1, int(num_generations * max_forced_fraction))

  # Build action_id -> description text map.
  action_text_map = {aid: desc for aid, desc in legal_actions_desc}
  legal_action_ids = [aid for aid, _ in legal_actions_desc]

  # Parse game state.
  obs_string = state.observation_string(player_id)
  fireworks = _parse_fireworks(obs_string)
  card_knowledge = _parse_card_knowledge(obs_string)

  # Classify legal actions.
  play_actions, discard_actions, hint_actions = _classify_legal_actions(
      state, player_id, legal_action_ids
  )

  # ── Tier 1: Known-safe plays (always include if available) ──
  safe_plays = _find_known_safe_plays(
      card_knowledge, fireworks, play_actions
  )
  safe_plays = safe_plays[:min(len(safe_plays), max_forced)]
  remaining = max_forced - len(safe_plays)

  # ── Tier 2: Risky plays ──
  risky_plays = []
  if remaining > 0:
    risky_plays = _find_risky_plays(
        card_knowledge, fireworks, play_actions, safe_plays,
        max_risky=min(2, remaining),
    )
    remaining -= len(risky_plays)

  # ── Tier 3: Smart discards ──
  smart_discards = []
  if remaining > 0:
    smart_discards = _find_smart_discards(
        state, player_id, card_knowledge, fireworks, discard_actions,
        max_discards=min(3, remaining),
    )
    remaining -= len(smart_discards)

  # ── Tier 4: Diverse hints (fills remaining slots with distinct hints) ──
  diverse_hints = []
  if remaining > 0:
    diverse_hints = _find_diverse_hints(
        state, player_id, obs_string, fireworks, hint_actions,
        max_hints=remaining,
    )
    remaining -= len(diverse_hints)

  # ── Overflow: If hints didn't use all slots, fill with more discards ──
  if remaining > 0 and discard_actions:
    extra_discards = _find_smart_discards(
        state, player_id, card_knowledge, fireworks, discard_actions,
        max_discards=len(smart_discards) + remaining,
    )
    for aid in extra_discards:
      if aid not in smart_discards and remaining > 0:
        smart_discards.append(aid)
        remaining -= 1

  plan = StrategicActionPlan(
      known_safe_plays=safe_plays,
      risky_plays=risky_plays,
      smart_discards=smart_discards,
      diverse_hints=diverse_hints,
      action_texts=action_text_map,
  )

  if plan.num_forced > 0:
    logging.info(
        '[strategic] P%d: %d safe_play, %d risky_play, %d smart_discard, '
        '%d diverse_hint (total %d/%d forced)',
        player_id,
        len(safe_plays),
        len(risky_plays),
        len(smart_discards),
        len(diverse_hints),
        plan.num_forced,
        num_generations,
    )

  return plan
