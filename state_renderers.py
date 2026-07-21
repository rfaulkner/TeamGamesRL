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

"""Text renderers for converting OpenSpiel game states to LLM prompts.

This module implements the 'text-action bridge' between OpenSpiel's discrete
game states and LLM text generation. Each renderer converts structured game
state into natural language prompts that an LLM can understand, and parses
the LLM's text output back into valid game actions.

The rendering pipeline is:

    OpenSpiel State  --(render_state)-->   Natural language prompt
    OpenSpiel Actions --(render_legal_actions)--> List of (id, text) pairs
    LLM text output  --(parse_action)-->   Action ID (int)

Supported games:
  - Negotiation: Multi-item deal-making with proposals and utterances.
  - Hanabi: Cooperative card game with imperfect information.
  - Generic: Fallback renderer using OpenSpiel's built-in string methods.

Action Encoding Summary (Negotiation):
  Actions are encoded as integers in two contiguous ranges:

  [0, NumDistinctProposals - 2]:
      Proposal actions. Each proposal is a vector [qty_item0, qty_item1, ...]
      encoded using mixed-radix encoding with base (kMaxQuantity + 1) = 6.
      For example, with 3 items, proposal [2, 1, 0] encodes as:
        2 * 6^2 + 1 * 6^1 + 0 * 6^0 = 72 + 6 + 0 = 78.

  NumDistinctProposals - 1:
      Special 'agreement' action — accept the most recent proposal.

  [NumDistinctProposals, NumDistinctProposals + NumDistinctUtterances - 1]:
      Utterance actions. Each utterance is a vector of symbol indices
      encoded similarly with base num_symbols.
"""

import abc
import difflib
import re
from typing import Optional

import pyspiel


# Maximum quantity per item type in the negotiation game.
_NEGOTIATION_MAX_QUANTITY = 5

# Human-readable names for item types, indexed by position.
# The negotiation game supports up to 5 item types (default 3).
_ITEM_NAMES = ['books', 'hats', 'balls', 'gems', 'cards']

# Hanabi color name mapping from single-letter codes to full names.
_HANABI_COLOR_NAMES = {
    'R': 'Red',
    'Y': 'Yellow',
    'G': 'Green',
    'B': 'Blue',
    'W': 'White',
}


class BaseStateRenderer(abc.ABC):
  """Abstract base class for converting game states to LLM-readable text.

  Subclasses implement game-specific rendering logic that transforms the
  structured OpenSpiel game state into natural language prompts suitable for
  LLM consumption. The three core methods form the text-action bridge:

    render_state:         State -> text prompt for the LLM.
    render_legal_actions: State -> list of (action_id, description) pairs.
    parse_action:         LLM text output -> best-matching action ID.
  """

  @abc.abstractmethod
  def render_state(
      self,
      state: pyspiel.State,
      player_id: int,
      game: pyspiel.Game,
  ) -> str:
    """Converts the current game state into a natural language prompt.

    Args:
      state: The current OpenSpiel game state.
      player_id: The ID of the player whose perspective to render from.
      game: The OpenSpiel game object (for accessing game parameters).

    Returns:
      A string prompt describing the game state for the LLM.
    """

  @abc.abstractmethod
  def render_legal_actions(
      self,
      state: pyspiel.State,
      player_id: int,
      game: pyspiel.Game,
  ) -> list[tuple[int, str]]:
    """Returns a list of (action_id, text_description) pairs for legal actions.

    Args:
      state: The current OpenSpiel game state.
      player_id: The ID of the player whose legal actions to render.
      game: The OpenSpiel game object.

    Returns:
      A list of tuples (action_id, human_readable_description).
    """

  @abc.abstractmethod
  def parse_action(
      self,
      text: str,
      legal_actions: list[tuple[int, str]],
  ) -> Optional[int]:
    """Parses the LLM's text output to find the best matching action ID.

    Uses fuzzy matching to map free-form LLM text to the closest legal
    action description.

    Args:
      text: The raw text output from the LLM.
      legal_actions: The list of (action_id, description) pairs from
        render_legal_actions.

    Returns:
      The action_id of the best matching action, or None if no reasonable
      match is found.
    """


def _fuzzy_match_action(
    text: str,
    legal_actions: list[tuple[int, str]],
    threshold: float = 0.3,
) -> Optional[int]:
  """Finds the best fuzzy match for LLM text among legal action descriptions.

  Uses difflib.SequenceMatcher to compute similarity ratios between the
  LLM's output text and each legal action description. Returns the action
  with the highest similarity, provided it exceeds the threshold.

  Args:
    text: The raw text output from the LLM, cleaned for matching.
    legal_actions: List of (action_id, description) pairs.
    threshold: Minimum similarity ratio to accept a match (0.0 to 1.0).

  Returns:
    The action_id of the best match, or None if no match exceeds threshold.
  """
  if not legal_actions:
    return None

  normalized_text = text.strip().lower()

  # First, try exact substring matching for efficiency.
  for action_id, description in legal_actions:
    normalized_desc = description.strip().lower()
    if normalized_desc in normalized_text or normalized_text in normalized_desc:
      return action_id

  # Fall back to fuzzy matching with SequenceMatcher.
  best_action_id = None
  best_ratio = threshold

  for action_id, description in legal_actions:
    normalized_desc = description.strip().lower()
    ratio = difflib.SequenceMatcher(
        None, normalized_text, normalized_desc
    ).ratio()
    if ratio > best_ratio:
      best_ratio = ratio
      best_action_id = action_id

  return best_action_id


def _decode_proposal(encoded: int, num_items: int) -> list[int]:
  """Decodes a proposal action ID into a quantity vector.

  Proposals are encoded using mixed-radix encoding with base
  (kMaxQuantity + 1) = 6 per item position. The encoding is big-endian:
  the first item is the most significant digit.

  For example, with 3 items and base 6:
    action_id 78 -> [2, 1, 0] (2*36 + 1*6 + 0*1 = 78)

  Args:
    encoded: The encoded proposal action ID.
    num_items: Number of distinct item types in the game.

  Returns:
    A list of quantities [qty_item0, qty_item1, ...].
  """
  base = _NEGOTIATION_MAX_QUANTITY + 1  # 6
  decoded = [0] * num_items
  for i in range(num_items - 1, -1, -1):
    decoded[i] = encoded % base
    encoded //= base
  return decoded


def _encode_proposal(quantities: list[int]) -> int:
  """Encodes a quantity vector into a proposal action ID.

  Inverse of _decode_proposal. Uses big-endian mixed-radix encoding
  with base 6.

  Args:
    quantities: List of quantities [qty_item0, qty_item1, ...].

  Returns:
    The encoded proposal action ID.
  """
  base = _NEGOTIATION_MAX_QUANTITY + 1  # 6
  encoded = 0
  for qty in quantities:
    encoded = encoded * base + qty
  return encoded


def _num_distinct_proposals(num_items: int) -> int:
  """Returns the total number of distinct proposal action IDs.

  This includes all possible quantity vectors plus one extra slot for the
  special 'agreement' action. The count is (kMaxQuantity + 1)^num_items + 1.

  Args:
    num_items: Number of distinct item types.

  Returns:
    Total number of proposal action slots (including agreement).
  """
  return (_NEGOTIATION_MAX_QUANTITY + 1) ** num_items + 1


class NegotiationRenderer(BaseStateRenderer):
  """Renders the negotiation game state as natural language.

  The negotiation game involves two players splitting a pool of items.
  Each round, a player proposes a split (how many of each item they want
  to take), optionally sends an utterance, and the other player can accept
  or counter-propose.

  Action encoding:
    - Actions [0, NumDistinctProposals - 2]: Proposal vectors encoded in
      mixed-radix base-6. Each position represents the quantity of an item
      type the proposing player wants to take.
    - Action NumDistinctProposals - 1: Agreement (accept the last proposal).
    - Actions [NumDistinctProposals, ...]: Utterances (symbolic communication
      channel), encoded similarly in base num_symbols.

  The renderer converts these numeric encodings into human-readable text
  like "Propose: take 2 books, 1 hat, 0 balls" and parses LLM responses
  back into action IDs.
  """

  def _get_game_params(
      self, game: pyspiel.Game
  ) -> tuple[int, bool, bool, int, int]:
    """Extracts negotiation-specific parameters from the game object.

    Args:
      game: The OpenSpiel game object.

    Returns:
      A tuple of (num_items, enable_proposals, enable_utterances,
        num_symbols, utterance_dim).
    """
    params = game.get_parameters()
    num_items = params.get('num_items', 3)
    enable_proposals = params.get('enable_proposals', True)
    enable_utterances = params.get('enable_utterances', True)
    num_symbols = params.get('num_symbols', 5)
    utterance_dim = params.get('utterance_dim', 3)
    return num_items, enable_proposals, enable_utterances, num_symbols, utterance_dim

  def _parse_observation_string(
      self, obs_str: str
  ) -> dict[str, str]:
    """Parses the raw observation string into structured components.

    The observation string from the negotiation game has the format:
      Max steps: N
      Item pool: q0 q1 q2
      Agent P util vec: v0 v1 v2
      Current player: P
      Turn Type: Proposal|Utterance
      [Most recent proposal: [p0, p1, p2]]
      [Most recent utterance: [u0, u1, u2]]

    Args:
      obs_str: The raw observation string from state.observation_string().

    Returns:
      A dict with keys: 'max_steps', 'item_pool', 'util_vec',
        'current_player', 'turn_type', 'recent_proposal', 'recent_utterance'.
    """
    result = {}
    for line in obs_str.strip().split('\n'):
      line = line.strip()
      if line.startswith('Max steps:'):
        result['max_steps'] = line.split(':')[1].strip()
      elif line.startswith('Item pool:'):
        result['item_pool'] = line.split(':')[1].strip()
      elif 'util vec:' in line:
        result['util_vec'] = line.split(':')[1].strip()
      elif line.startswith('Current player:'):
        result['current_player'] = line.split(':')[1].strip()
      elif line.startswith('Turn Type:'):
        result['turn_type'] = line.split(':')[1].strip()
      elif line.startswith('Most recent proposal:'):
        result['recent_proposal'] = line.split(':')[1].strip()
      elif line.startswith('Most recent utterance:'):
        result['recent_utterance'] = line.split(':')[1].strip()
    return result

  def _format_item_list(
      self, quantities: list[int], num_items: int
  ) -> str:
    """Formats a quantity vector as a human-readable item list.

    Args:
      quantities: List of quantities per item type.
      num_items: Number of item types.

    Returns:
      A string like "3 books, 2 hats, 4 balls".
    """
    parts = []
    for i in range(min(len(quantities), num_items)):
      name = _ITEM_NAMES[i] if i < len(_ITEM_NAMES) else f'item_{i}'
      parts.append(f'{quantities[i]} {name}')
    return ', '.join(parts)

  def _format_value_list(
      self, values: list[int], num_items: int
  ) -> str:
    """Formats a value vector as human-readable item values.

    Args:
      values: List of utility values per item type.
      num_items: Number of item types.

    Returns:
      A string like "books=5, hats=3, balls=1".
    """
    parts = []
    for i in range(min(len(values), num_items)):
      name = _ITEM_NAMES[i] if i < len(_ITEM_NAMES) else f'item_{i}'
      parts.append(f'{name}={values[i]}')
    return ', '.join(parts)

  def render_state(
      self,
      state: pyspiel.State,
      player_id: int,
      game: pyspiel.Game,
  ) -> str:
    """Renders the negotiation state as a natural language prompt.

    Produces output like:
      You are Player 0, negotiating a deal with another player.
      Item pool: 3 books, 2 hats, 4 balls
      Your values: books=5, hats=3, balls=1
      [Previous proposals...]
      It is your turn to make a proposal.

    Args:
      state: The current OpenSpiel game state.
      player_id: The ID of the player to render for.
      game: The OpenSpiel game object.

    Returns:
      A natural language prompt string.
    """
    num_items, _, enable_utterances, _, _ = self._get_game_params(game)

    obs_str = state.observation_string(player_id)
    parsed = self._parse_observation_string(obs_str)

    # Build the prompt.
    lines = []
    lines.append(
        f'You are Player {player_id}, negotiating a deal with another player.'
    )

    # Item pool.
    if 'item_pool' in parsed:
      pool_quantities = [int(x) for x in parsed['item_pool'].split()]
      pool_str = self._format_item_list(pool_quantities, num_items)
      lines.append(f'Item pool: {pool_str}')

    # Player's own utility values.
    if 'util_vec' in parsed:
      values = [int(x) for x in parsed['util_vec'].split()]
      values_str = self._format_value_list(values, num_items)
      lines.append(f'Your values: {values_str}')

    # Show proposal history from the full game state string.
    full_str = state.to_string()
    proposal_lines = []
    for line in full_str.split('\n'):
      if 'proposes:' in line:
        # Parse "Player P proposes: [q0, q1, q2]" into natural language.
        match = re.search(
            r'Player (\d+) proposes: \[([^\]]+)\]', line
        )
        if match:
          proposer = int(match.group(1))
          raw_quantities = [int(x.strip()) for x in match.group(2).split(',')]
          proposal_text = self._format_item_list(raw_quantities, num_items)
          role = 'You' if proposer == player_id else 'Opponent'
          proposal_lines.append(
              f'  {role} proposed to take: {proposal_text}'
          )
    if proposal_lines:
      lines.append('Previous proposals:')
      lines.extend(proposal_lines)

    # Current turn type.
    turn_type = parsed.get('turn_type', 'Proposal').strip()
    if turn_type == 'Proposal':
      if 'recent_proposal' in parsed:
        lines.append(
            'It is your turn to make a counter-proposal or accept the deal.'
        )
      else:
        lines.append('It is your turn to make a proposal.')
    elif turn_type == 'Utterance':
      lines.append('It is your turn to send a message (utterance).')

    return '\n'.join(lines)

  def render_legal_actions(
      self,
      state: pyspiel.State,
      player_id: int,
      game: pyspiel.Game,
  ) -> list[tuple[int, str]]:
    """Renders legal actions as human-readable (action_id, description) pairs.

    Proposal actions are rendered as:
      "Propose: take 2 books, 1 hat, 0 balls"
    The agreement action is rendered as:
      "Accept the current proposal"
    Utterance actions are rendered as:
      "Send utterance: [symbol0, symbol1, symbol2]"

    Args:
      state: The current OpenSpiel game state.
      player_id: The ID of the player.
      game: The OpenSpiel game object.

    Returns:
      A list of (action_id, description) tuples.
    """
    num_items, _, _, _, _ = self._get_game_params(game)
    num_proposals = _num_distinct_proposals(num_items)
    agreement_action = num_proposals - 1

    legal_action_ids = state.legal_actions(player_id)
    result = []

    for action_id in legal_action_ids:
      if action_id == agreement_action:
        # Special agreement action.
        result.append((action_id, 'Accept the current proposal'))
      elif action_id < num_proposals:
        # Regular proposal action: decode and render as item list.
        quantities = _decode_proposal(action_id, num_items)
        item_str = self._format_item_list(quantities, num_items)
        result.append((action_id, f'Propose: take {item_str}'))
      else:
        # Utterance action: show the raw action string from the game.
        action_str = state.action_to_string(player_id, action_id)
        # Clean up the OpenSpiel format ", Utterance: [s0, s1, s2]".
        action_str = action_str.strip().lstrip(',').strip()
        result.append((action_id, f'Send {action_str.lower()}'))

    return result

  def parse_action(
      self,
      text: str,
      legal_actions: list[tuple[int, str]],
  ) -> Optional[int]:
    """Parses LLM text output to find the best matching negotiation action.

    First tries to detect specific patterns like "accept" or quantity-based
    proposals, then falls back to fuzzy matching.

    Args:
      text: The raw text output from the LLM.
      legal_actions: List of (action_id, description) pairs.

    Returns:
      The best matching action_id, or None if no match found.
    """
    normalized = text.strip().lower()

    # Check for acceptance keywords first.
    if any(kw in normalized for kw in ['accept', 'agree', 'deal']):
      for action_id, desc in legal_actions:
        if 'accept' in desc.lower():
          return action_id

    # Try to parse a proposal pattern like "take 2 books, 1 hat, 0 balls".
    # Look for quantity-item patterns.
    quantity_pattern = re.compile(r'(\d+)\s+(' + '|'.join(_ITEM_NAMES) + r')')
    matches = quantity_pattern.findall(normalized)
    if matches:
      # Build a lookup from item name to quantity.
      parsed_quantities = {name: int(qty) for qty, name in matches}
      # Find the proposal that matches these quantities.
      for action_id, desc in legal_actions:
        desc_matches = quantity_pattern.findall(desc.lower())
        if desc_matches:
          desc_quantities = {name: int(qty) for qty, name in desc_matches}
          if parsed_quantities == desc_quantities:
            return action_id

    # Fall back to generic fuzzy matching.
    return _fuzzy_match_action(text, legal_actions)


class HanabiRenderer(BaseStateRenderer):
  """Renders Hanabi game states as natural language for LLM agents.

  Hanabi is a cooperative card game where players can see others' hands but
  not their own. Players must work together to play cards in the correct
  order (rank 1 through max rank) for each color (suit).

  The renderer translates the Hanabi observation string into a readable
  description of:
    - Game status (life tokens, information tokens, fireworks progress).
    - Other players' visible hands.
    - The current player's known card information from hints.
    - Available actions (play, discard, hint about color/rank).

  OpenSpiel Hanabi action format:
    Actions are encoded by the underlying hanabi_learning_env and exposed
    via ActionToString as one of:
      "(Discard N)"       - Discard card at position N from hand.
      "(Play N)"          - Play card at position N from hand.
      "(Reveal player +X color C)" - Hint to player +X about color C.
      "(Reveal player +X rank R)"  - Hint to player +X about rank R.
  """

  def _parse_hanabi_observation(
      self, obs_str: str
  ) -> dict[str, object]:
    """Parses the raw Hanabi observation string into structured data.

    The observation string has the format:
      Life tokens: N
      Info tokens: N
      Fireworks: R0 Y0 G0 B0 W0
      Hands:
      [Cur player | card_info]
      [Card || Knowledge]
      ...
      -----
      ...
      Deck size: N
      Discards: [cards]

    Args:
      obs_str: Raw observation string from state.observation_string().

    Returns:
      A dict with keys: 'life_tokens', 'info_tokens', 'fireworks',
        'hands' (list of hand info dicts), 'deck_size', 'discards'.
    """
    result = {
        'life_tokens': 0,
        'info_tokens': 0,
        'fireworks': '',
        'hands': [],
        'deck_size': 0,
        'discards': '',
    }

    lines = obs_str.strip().split('\n')
    current_hand = None
    is_current_player = False

    for line in lines:
      line = line.strip()

      if line.startswith('Life tokens:'):
        result['life_tokens'] = int(line.split(':')[1].strip())
      elif line.startswith('Info tokens:'):
        result['info_tokens'] = int(line.split(':')[1].strip())
      elif line.startswith('Fireworks:'):
        result['fireworks'] = line.split(':')[1].strip()
      elif line.startswith('Deck size:'):
        result['deck_size'] = int(line.split(':')[1].strip())
      elif line.startswith('Discards:'):
        result['discards'] = line.split(':')[1].strip()
      elif line == 'Hands:':
        continue
      elif line == '-----':
        # Separator between hands. Save current hand if any.
        if current_hand is not None:
          result['hands'].append(current_hand)
        current_hand = None
        is_current_player = False
      elif line == 'Cur player':
        is_current_player = True
        current_hand = {
            'is_current_player': True,
            'cards': [],
        }
      elif '||' in line or line.startswith('XX') or line.startswith('X'):
        # Card line: "R2 || XX|RY123" or "XX || XX|RY123"
        if current_hand is None:
          current_hand = {
              'is_current_player': is_current_player,
              'cards': [],
          }
        current_hand['cards'].append(line)

    # Don't forget the last hand.
    if current_hand is not None:
      result['hands'].append(current_hand)

    return result

  def _format_fireworks(self, fireworks_str: str) -> str:
    """Formats fireworks string into readable text.

    Converts "R0 Y2 G1" into "Red: 0, Yellow: 2, Green: 1".

    Args:
      fireworks_str: Raw fireworks string like "R0 Y2 G1".

    Returns:
      Human-readable fireworks status.
    """
    if not fireworks_str.strip():
      return 'None yet'

    parts = []
    for token in fireworks_str.strip().split():
      # Token format: "R0", "Y2", etc.
      if len(token) >= 2:
        color_code = token[0]
        rank = token[1:]
        color_name = _HANABI_COLOR_NAMES.get(color_code, color_code)
        parts.append(f'{color_name}: {rank}')
    return ', '.join(parts) if parts else fireworks_str

  def _format_card_info(
      self, card_str: str, is_own_hand: bool
  ) -> str:
    """Formats a single card's information into readable text.

    Card strings have the format:
      "R2 || XX|RY123" - visible card R2, knowledge is unknown color/all ranks
      "XX || X2|RY2"   - unknown card, knowledge is unknown color/rank 2

    The "||" separates the actual card from the knowledge. For the current
    player's own hand, the actual card is "XX" (unknown).

    Args:
      card_str: Raw card string like "R2 || XX|RY123".
      is_own_hand: Whether this card belongs to the current player.

    Returns:
      Human-readable card description.
    """
    parts = card_str.split('||')
    if len(parts) != 2:
      return card_str.strip()

    actual = parts[0].strip()
    knowledge = parts[1].strip()

    if is_own_hand:
      # For own hand, describe what we know from hints.
      return self._describe_card_knowledge(knowledge)
    else:
      # For other players, show the actual card.
      return self._describe_visible_card(actual, knowledge)

  def _describe_visible_card(
      self, actual: str, knowledge: str
  ) -> str:
    """Describes a visible card (in another player's hand).

    Args:
      actual: The actual card identity, e.g. "R2", "Y1".
      knowledge: The card knowledge string, e.g. "XX|RY123".

    Returns:
      Human-readable description like "Red 2".
    """
    if len(actual) >= 2 and actual[0] in _HANABI_COLOR_NAMES:
      color = _HANABI_COLOR_NAMES[actual[0]]
      rank = actual[1:]
      return f'{color} {rank}'
    return actual

  def _describe_card_knowledge(self, knowledge: str) -> str:
    """Describes what is known about a card from hints.

    Knowledge format: "color_info|rank_info"
    Examples:
      "XX|RY123" -> unknown color, could be ranks 1,2,3
      "X2|RY2"   -> unknown color, known rank 2
      "R2|R2"    -> known Red 2

    The knowledge part before "|" encodes color info, and after "|"
    encodes rank info. "X" in each position means unknown.

    Args:
      knowledge: The knowledge string like "XX|RY123" or "X2|RY2".

    Returns:
      Human-readable description of known card information.
    """
    parts = knowledge.split('|')
    if len(parts) != 2:
      return f'Unknown (raw: {knowledge})'

    card_part = parts[0]  # e.g., "XX", "X2", "R2"
    hint_part = parts[1]  # e.g., "RY123", "RY2", "R2"

    # Determine known color.
    known_color = None
    possible_colors = []
    if len(card_part) >= 1 and card_part[0] != 'X':
      known_color = _HANABI_COLOR_NAMES.get(card_part[0], card_part[0])
    else:
      # Extract possible colors from hint_part (letters at the start).
      for ch in hint_part:
        if ch.isalpha() and ch in _HANABI_COLOR_NAMES:
          possible_colors.append(_HANABI_COLOR_NAMES[ch])
        elif ch.isdigit():
          break

    # Determine known rank.
    known_rank = None
    possible_ranks = []
    if len(card_part) >= 2 and card_part[1] != 'X':
      known_rank = card_part[1]
    else:
      # Extract possible ranks from hint_part (digits).
      for ch in hint_part:
        if ch.isdigit():
          possible_ranks.append(ch)

    # Build description.
    desc_parts = []
    if known_color:
      desc_parts.append(f'Color: {known_color}')
    elif possible_colors:
      desc_parts.append(f'Possible colors: {", ".join(possible_colors)}')
    else:
      desc_parts.append('Color: unknown')

    if known_rank:
      desc_parts.append(f'Rank: {known_rank}')
    elif possible_ranks:
      desc_parts.append(f'Possible ranks: {", ".join(possible_ranks)}')
    else:
      desc_parts.append('Rank: unknown')

    return '; '.join(desc_parts)

  def render_state(
      self,
      state: pyspiel.State,
      player_id: int,
      game: pyspiel.Game,
  ) -> str:
    """Renders the Hanabi game state as a natural language prompt.

    Args:
      state: The current OpenSpiel game state.
      player_id: The ID of the player to render for.
      game: The OpenSpiel game object.

    Returns:
      A natural language prompt describing the Hanabi game state.
    """
    obs_str = state.observation_string(player_id)
    parsed = self._parse_hanabi_observation(obs_str)
    num_players = game.num_players()

    lines = []
    lines.append(f'You are Player {player_id} in a {num_players}-player'
                 ' Hanabi game.')
    lines.append(f'Life tokens remaining: {parsed["life_tokens"]}')
    lines.append(f'Information tokens remaining: {parsed["info_tokens"]}')
    lines.append(
        f'Fireworks on table: {self._format_fireworks(parsed["fireworks"])}'
    )
    lines.append(f'Cards remaining in deck: {parsed["deck_size"]}')

    if parsed['discards']:
      lines.append(f'Discarded cards: {parsed["discards"]}')
    else:
      lines.append('No cards discarded yet.')

    # Render each player's hand.
    # The observation orders hands relative to the current player.
    # The hands list contains entries for each visible player.
    lines.append('')
    hand_idx = 0
    for hand_info in parsed['hands']:
      if hand_info['is_current_player']:
        lines.append('Your hand (you cannot see your own cards):')
        for i, card_str in enumerate(hand_info['cards']):
          card_desc = self._format_card_info(card_str, is_own_hand=True)
          lines.append(f'  Card {i}: {card_desc}')
      else:
        hand_idx += 1
        # Determine the actual player ID for this hand.
        # Hands are displayed in order relative to current player.
        other_player = (player_id + hand_idx) % num_players
        lines.append(f"Player {other_player}'s hand:")
        for i, card_str in enumerate(hand_info['cards']):
          card_desc = self._format_card_info(card_str, is_own_hand=False)
          lines.append(f'  Card {i}: {card_desc}')

    current = state.current_player()
    if current == player_id:
      lines.append('')
      lines.append('It is your turn. Choose an action.')

    return '\n'.join(lines)

  def render_legal_actions(
      self,
      state: pyspiel.State,
      player_id: int,
      game: pyspiel.Game,
  ) -> list[tuple[int, str]]:
    """Renders Hanabi legal actions as human-readable descriptions.

    Translates OpenSpiel action strings into natural language:
      "(Discard 0)"               -> "Discard card 0 from your hand"
      "(Play 1)"                  -> "Play card 1 from your hand"
      "(Reveal player +1 color R)" -> "Hint Player 2 about Red cards"
      "(Reveal player +1 rank 2)" -> "Hint Player 2 about rank 2 cards"

    Args:
      state: The current OpenSpiel game state.
      player_id: The ID of the player.
      game: The OpenSpiel game object.

    Returns:
      A list of (action_id, description) tuples.
    """
    num_players = game.num_players()
    legal_action_ids = state.legal_actions(player_id)
    result = []

    for action_id in legal_action_ids:
      action_str = state.action_to_string(player_id, action_id)

      # Parse the OpenSpiel action string format.
      discard_match = re.match(r'\(Discard (\d+)\)', action_str)
      play_match = re.match(r'\(Play (\d+)\)', action_str)
      reveal_color_match = re.match(
          r'\(Reveal player \+(\d+) color (\w+)\)', action_str
      )
      reveal_rank_match = re.match(
          r'\(Reveal player \+(\d+) rank (\d+)\)', action_str
      )

      if discard_match:
        card_idx = discard_match.group(1)
        desc = f'Discard card {card_idx} from your hand'
      elif play_match:
        card_idx = play_match.group(1)
        desc = f'Play card {card_idx} from your hand'
      elif reveal_color_match:
        offset = int(reveal_color_match.group(1))
        color_code = reveal_color_match.group(2)
        target_player = (player_id + offset) % num_players
        color_name = _HANABI_COLOR_NAMES.get(color_code, color_code)
        desc = f'Hint Player {target_player} about {color_name} cards'
      elif reveal_rank_match:
        offset = int(reveal_rank_match.group(1))
        rank = reveal_rank_match.group(2)
        target_player = (player_id + offset) % num_players
        desc = f'Hint Player {target_player} about rank {rank} cards'
      else:
        # Fallback: use the raw action string.
        desc = action_str

      result.append((action_id, desc))

    return result

  def parse_action(
      self,
      text: str,
      legal_actions: list[tuple[int, str]],
  ) -> Optional[int]:
    """Parses LLM text output to find the best matching Hanabi action.

    Tries pattern-based matching first (looking for keywords like "play",
    "discard", "hint"), then falls back to fuzzy matching.

    Args:
      text: The raw text output from the LLM.
      legal_actions: List of (action_id, description) pairs.

    Returns:
      The best matching action_id, or None if no match found.
    """
    normalized = text.strip().lower()

    # Try keyword-based matching first for common action patterns.
    # "play card 2" or "play 2"
    play_match = re.search(r'\bplay\b.*?(\d+)', normalized)
    if play_match:
      card_idx = play_match.group(1)
      for action_id, desc in legal_actions:
        if f'play card {card_idx}' in desc.lower():
          return action_id

    # "discard card 1" or "discard 1"
    discard_match = re.search(r'\bdiscard\b.*?(\d+)', normalized)
    if discard_match:
      card_idx = discard_match.group(1)
      for action_id, desc in legal_actions:
        if f'discard card {card_idx}' in desc.lower():
          return action_id

    # "hint player 2 about red" or "tell player 2 about rank 3"
    hint_match = re.search(
        r'(?:hint|tell|reveal).*?player\s+(\d+).*?(?:about\s+)?(\w+)',
        normalized,
    )
    if hint_match:
      target = hint_match.group(1)
      subject = hint_match.group(2)
      for action_id, desc in legal_actions:
        desc_lower = desc.lower()
        if f'player {target}' in desc_lower and subject in desc_lower:
          return action_id

    # Fall back to fuzzy matching.
    return _fuzzy_match_action(text, legal_actions)


class GenericRenderer(BaseStateRenderer):
  """Fallback renderer using OpenSpiel's built-in string representations.

  This renderer works with any OpenSpiel game by using the game's native
  ToString() and ActionToString() methods. While less readable than
  game-specific renderers, it provides a universal baseline.
  """

  def render_state(
      self,
      state: pyspiel.State,
      player_id: int,
      game: pyspiel.Game,
  ) -> str:
    """Renders the game state using OpenSpiel's built-in ToString().

    Prepends player context information to the raw state string.

    Args:
      state: The current OpenSpiel game state.
      player_id: The ID of the player to render for.
      game: The OpenSpiel game object.

    Returns:
      A string containing the game state description.
    """
    lines = [
        f'Game: {game.get_type().long_name}',
        f'You are Player {player_id} of {game.num_players()}.',
    ]

    # Use observation_string if available for player-specific view,
    # otherwise fall back to the full state string.
    try:
      obs_str = state.observation_string(player_id)
      lines.append(f'Your observation:\n{obs_str}')
    except (RuntimeError, pyspiel.SpielError):
      lines.append(f'Game state:\n{state}')

    if state.current_player() == player_id:
      lines.append('\nIt is your turn. Choose an action from the list.')

    return '\n'.join(lines)

  def render_legal_actions(
      self,
      state: pyspiel.State,
      player_id: int,
      game: pyspiel.Game,
  ) -> list[tuple[int, str]]:
    """Renders legal actions using OpenSpiel's ActionToString().

    Args:
      state: The current OpenSpiel game state.
      player_id: The ID of the player.
      game: The OpenSpiel game object.

    Returns:
      A list of (action_id, description) tuples.
    """
    legal_action_ids = state.legal_actions(player_id)
    return [
        (action_id, state.action_to_string(player_id, action_id))
        for action_id in legal_action_ids
    ]

  def parse_action(
      self,
      text: str,
      legal_actions: list[tuple[int, str]],
  ) -> Optional[int]:
    """Parses LLM text using fuzzy matching against action strings.

    Also checks if the LLM output contains a raw action ID number.

    Args:
      text: The raw text output from the LLM.
      legal_actions: List of (action_id, description) pairs.

    Returns:
      The best matching action_id, or None if no match found.
    """
    # First, check if the text contains a raw action ID.
    stripped = text.strip()
    try:
      raw_id = int(stripped)
      for action_id, _ in legal_actions:
        if action_id == raw_id:
          return raw_id
    except ValueError:
      pass

    # Fall back to fuzzy matching.
    return _fuzzy_match_action(text, legal_actions)


def get_renderer(game_name: str) -> BaseStateRenderer:
  """Factory function to get the appropriate renderer for a game.

  Args:
    game_name: The short name of the OpenSpiel game (e.g., 'negotiation',
      'hanabi').

  Returns:
    An instance of the appropriate BaseStateRenderer subclass.
  """
  renderers = {
      'negotiation': NegotiationRenderer,
      'hanabi': HanabiRenderer,
  }

  renderer_class = renderers.get(game_name, GenericRenderer)
  return renderer_class()
