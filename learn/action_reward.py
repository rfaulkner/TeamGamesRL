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

"""Dense per-action reward shaping for Hanabi.

Instead of rolling the game out to completion and using the terminal score
as reward, this module evaluates the *immediate quality* of each action:

  - **Play**: +1.0 for a valid play (card adds to fireworks), -1.0 for an
    invalid play (life token lost), +1.5 if the play completes a colour
    stack (rank 5, which also regains an info token).
  - **Discard**: Ranges from +0.3 (dead card = already played or duplicate)
    to -1.0 (discarding a hinted card known to be playable).  Intermediate
    penalties apply for discarding last copies of needed cards (-0.5 to
    -0.8 depending on hint status) and partially-playable hinted cards
    (-0.3).  Discarding unhinted non-critical cards gives +0.1.
  - **Hint**: Reward scales with the information gain delivered to the
    partner, with a bonus if the hint enables an immediate play.
  - **Parse failure**: -0.3 penalty when the LLM output cannot be parsed
    into a valid Hanabi action.

The rewards are *incremental contributions to the Hanabi score* -- they
reflect how much each action moves the team toward (or away from) a
higher final score.

Usage::

  from learn.action_reward import evaluate_action_quality
  reward = evaluate_action_quality(state, action_id, player_id)

For chained dense evaluation over a short rollout::

  from learn.action_reward import evaluate_dense_chain
  total = evaluate_dense_chain(runner, state, first_action, player_id,
                               horizon=4, discount=0.9)
"""

from __future__ import annotations

from typing import Optional

from absl import logging
from env.hanabi.hanabi_env import CARD_KNOWLEDGE_RE as _CARD_KNOWLEDGE_RE
from env.hanabi.hanabi_env import COLOR_CHARS as _COLOR_CHARS
from env.hanabi.hanabi_env import deserialize_game_and_state
from env.hanabi.hanabi_env import DISCARD_RE as _DISCARD_RE
from env.hanabi.hanabi_env import FIREWORKS_RE as _FIREWORKS_RE
from env.hanabi.hanabi_env import parse_fireworks
from env.hanabi.hanabi_env import PLAY_RE as _PLAY_RE
from env.hanabi.hanabi_env import REVEAL_COLOR_RE as _REVEAL_COLOR_RE
from env.hanabi.hanabi_env import REVEAL_RANK_RE as _REVEAL_RANK_RE
from env.hanabi.hanabi_env import VISIBLE_CARD_RE as _VISIBLE_CARD_RE
import numpy as np

# -- Constants ----------------------------------------------------------------

# Maximum ranks per colour in standard Hanabi.
_MAX_RANK = 5
# Card multiplicities by rank in standard Hanabi (rank 1-indexed).
# Rank 1: 3 copies, Ranks 2-4: 2 copies, Rank 5: 1 copy.
_RANK_COPIES = {1: 3, 2: 2, 3: 2, 4: 2, 5: 1}


# =============================================================================
# Core reward function
# =============================================================================


def evaluate_action_quality(
    state,
    action_id: int,
    player_id: int,
    game=None,
) -> float:
  """Compute the immediate reward for a Hanabi action.

  Evaluates the quality of ``action_id`` applied to ``state`` by the
  acting ``player_id``.  The reward is computed from the state
  transition without any forward simulation.

  Args:
    state: A ``HanabiState`` (or OpenSpiel state) supporting
        ``score()``, ``life_tokens()``, ``information_tokens()``,
        ``observation_string()``, ``action_to_string()``, ``clone()``,
        and ``apply_action()``.
    action_id: The action UID to evaluate.
    player_id: The index of the acting player.
    game: Optional game object (unused, kept for API compatibility).

  Returns:
    A float reward in approximately [-1.0, +1.5].
  """
  del game  # Unused.
  action_str = state.action_to_string(player_id, action_id)

  # -- Play action ----------------------------------------------------------
  play_match = _PLAY_RE.search(action_str)
  if play_match:
    card_pos = int(play_match.group(1))
    return _evaluate_play(state, action_id, player_id, card_pos)

  # -- Discard action -------------------------------------------------------
  discard_match = _DISCARD_RE.search(action_str)
  if discard_match:
    card_pos = int(discard_match.group(1))
    return _evaluate_discard(state, action_id, player_id, card_pos)

  # -- Hint action ----------------------------------------------------------
  color_match = _REVEAL_COLOR_RE.search(action_str)
  if color_match:
    target_offset = int(color_match.group(1))
    return _evaluate_hint(
        state, player_id, target_offset, 'color', color_match.group(2)
    )

  rank_match = _REVEAL_RANK_RE.search(action_str)
  if rank_match:
    target_offset = int(rank_match.group(1))
    return _evaluate_hint(
        state, player_id, target_offset, 'rank', rank_match.group(2)
    )

  # Unknown action type -- neutral.
  logging.warning('Unknown action type: %s', action_str)
  return 0.0


# =============================================================================
# Playability and play evaluation
# =============================================================================


def _card_playability_from_knowledge(
    card_knowledge: list[tuple[str, str]],
    card_pos: int,
    fireworks: dict[str, int],
) -> tuple[bool, float]:
  """Check if a card could be playable based on the player's own knowledge.

  Uses the player's card knowledge (from hints) to assess whether the
  card at ``card_pos`` might be playable.  This is the player's
  *subjective* view -- they can't see the card, only what hints have
  told them.

  Args:
    card_knowledge: Per-card ``(possible_colors, possible_ranks)``
        tuples from the player's observation.
    card_pos: The card's position in hand (0-indexed).
    fireworks: Current firework heights per colour.

  Returns:
    A tuple of ``(is_hinted, playable_fraction)``:
      - ``is_hinted``: True if the card has received at least one hint
        (i.e., knowledge is narrower than the full 5 colors × 5 ranks).
      - ``playable_fraction``: Fraction of remaining (color, rank)
        possibilities that would be immediately playable.  0.0 if
        unhinted or no playable possibilities.
  """
  if card_pos >= len(card_knowledge):
    return False, 0.0

  colors, ranks = card_knowledge[card_pos]

  # Unhinted = all 5 colors and all 5 ranks still possible.
  is_hinted = not (len(colors) >= 5 and len(ranks) >= 5)

  if not is_hinted:
    return False, 0.0

  # Count how many (color, rank) combos are playable.
  total_combos = len(colors) * len(ranks)
  if total_combos == 0:
    return True, 0.0

  playable_combos = 0
  for c in colors:
    needed_rank = fireworks.get(c, 0) + 1
    for r in ranks:
      if int(r) == needed_rank:
        playable_combos += 1

  return True, playable_combos / total_combos


def _evaluate_play(
    state,
    action_id: int,
    player_id: int = 0,
    card_pos: int = 0,
) -> float:
  """Evaluate a Play action by checking the state transition.

  Clones the state, applies the action, and checks whether the score
  increased (successful play) or a life token was lost (failed play).
  If the card was hinted with positive playability potential, the
  failure penalty is softened to prevent play extinction.

  Args:
    state: The pre-action state.
    action_id: The Play action UID.
    player_id: Index of the acting player.
    card_pos: Hand position of the played card.

  Returns:
    +1.0 for a valid play, +1.5 if it completes a colour, -0.3 for
    a failed play on a hinted card, -1.0 for an unhinted blind failed play.
  """
  score_before = state.score()
  lives_before = state.life_tokens()

  sim = state.clone()
  sim.apply_action(action_id)

  score_after = sim.score()
  lives_after = sim.life_tokens()

  if score_after > score_before:
    # Successful play.
    # Check if this completed a colour stack (rank 5 -> score mod 5 == 0
    # after increment, meaning the stack went from 4 to 5).
    score_delta = score_after - score_before
    if score_delta == 1 and score_after % _MAX_RANK == 0:
      return 1.5  # Completed a colour + regains info token.
    return 1.0
  elif lives_after < lives_before:
    # Invalid play -- lost a life token.
    # Check if the player had clues suggesting playability.
    obs_str = state.observation_string(player_id)
    card_knowledge = _CARD_KNOWLEDGE_RE.findall(obs_str)
    fireworks = _parse_fireworks_from_state(state, player_id)
    is_hinted, playable_frac = _card_playability_from_knowledge(
        card_knowledge, card_pos, fireworks
    )
    if is_hinted and playable_frac > 0:
      return -0.3  # Softened penalty for play attempt on hinted card.
    return -1.0  # Full penalty for unhinted blind play.
  else:
    # Edge case: score didn't change, no life lost (shouldn't happen
    # in standard Hanabi, but handle gracefully).
    return 0.0


# =============================================================================
# Discard evaluation
# =============================================================================


def _evaluate_discard(
    state,
    action_id: int,
    player_id: int,
    card_pos: int,
) -> float:
  """Evaluate a Discard action by checking the card's strategic value.

  Uses the game state and player observation to determine:
    - Is the card already played (dead)? -> safe discard (+0.35 + token bonus).
    - Was it hinted and known playable? -> severe penalty (-0.8).
    - Was it hinted and the last copy? -> critical loss (-0.6).
    - Was it unhinted? -> routine discard to regain clues (+0.15 to +0.30),
      with token scarcity bonus. The player cannot see unhinted cards,
      so unhinted discards are not heavily penalized for hidden playable status.

  Args:
    state: The pre-action state.
    action_id: The Discard action UID.
    player_id: The acting player.
    card_pos: The card's position in hand (0-indexed).

  Returns:
    A reward in [-0.8, +0.5].
  """
  info_tokens = state.information_tokens()
  # Regaining info tokens is more valuable when the team is clue-starved.
  if info_tokens <= 2:
    token_bonus = 0.15
  elif info_tokens <= 5:
    token_bonus = 0.05
  else:
    token_bonus = 0.0

  fireworks = _parse_fireworks_from_state(state, player_id)

  sim = state.clone()
  sim.apply_action(action_id)

  # ── Knowledge-aware penalty ──
  obs_str = state.observation_string(player_id)
  card_knowledge = _CARD_KNOWLEDGE_RE.findall(obs_str)
  is_hinted, playable_frac = _card_playability_from_knowledge(
      card_knowledge, card_pos, fireworks
  )

  discarded_card = _identify_discarded_card(state, sim, player_id)
  if discarded_card is None:
    return 0.10 + token_bonus

  card_color, card_rank = discarded_card

  # Case 1: Card is already played (rank <= firework height).
  fw_height = fireworks.get(card_color, 0)
  if card_rank <= fw_height:
    return 0.35 + token_bonus  # Dead card -- excellent safe discard.

  # Case 2: Card was HINTED (player had information about this card).
  if is_hinted:
    if playable_frac >= 0.5:
      # Player had strong hints suggesting playability -- severe penalty.
      return -0.8
    remaining = _count_remaining_copies(
        state, sim, card_color, card_rank, player_id
    )
    if remaining == 0:
      # Hinted last copy -- team invested info tokens, critical loss.
      return -0.6
    if playable_frac > 0:
      # Hinted with playable potential.
      return -0.25
    # Hinted, but known not playable right now.
    return 0.10 + token_bonus

  # Case 3: Card was UNHINTED (player had no knowledge of this card).
  # In Hanabi, players must discard unhinted cards to regain info tokens.
  # Do not severely penalize the player for hidden state they cannot observe.
  remaining = _count_remaining_copies(
      state, sim, card_color, card_rank, player_id
  )
  if remaining == 0:
    # Unhinted last copy lost. Unfortunate, but player had no information.
    return -0.15 if info_tokens > 2 else -0.05

  if card_rank == fw_height + 1:
    # Unhinted card happened to be currently playable.
    # When tokens are low (<= 2), discard was necessary to unblock the team.
    return 0.10 if info_tokens <= 2 else 0.0

  # Routine unhinted discard with remaining copies.
  return 0.15 + token_bonus


def _parse_fireworks_from_state(state, player_id: int) -> dict[str, int]:
  """Extract firework heights from the state's observation string."""
  return parse_fireworks(state.observation_string(player_id))


def _identify_discarded_card(
    state_before,
    state_after,
    player_id: int,
) -> Optional[tuple[str, int]]:
  """Identify the card that was discarded by diffing discard piles.

  Compares the discard section of the observation strings before and
  after the discard action to find the newly added card.

  Args:
    state_before: State before the discard.
    state_after: State after the discard.
    player_id: The acting player (for observation perspective).

  Returns:
    A (color_letter, rank_int) tuple, or None if identification fails.
  """
  def _parse_discards(obs: str) -> list[str]:
    for line in obs.split('\n'):
      if line.strip().startswith('Discards:'):
        cards_str = line.split(':', 1)[1].strip()
        if cards_str:
          return cards_str.split()
    return []

  # Use the acting player's perspective to see the full discard pile.
  before_obs = state_before.observation_string(player_id)
  after_obs = state_after.observation_string(player_id)

  before_discards = _parse_discards(before_obs)
  after_discards = _parse_discards(after_obs)

  # The new card is any card in after but not in before.
  if len(after_discards) > len(before_discards):
    new_card_str = after_discards[-1]  # Most recent discard.
    if len(new_card_str) >= 2 and new_card_str[0] in _COLOR_CHARS:
      return (new_card_str[0], int(new_card_str[1:]))

  return None


def _count_remaining_copies(
    state_before,
    state_after,
    color: str,
    rank: int,
    player_id: int,
) -> int:
  """Count remaining copies of a card after a discard.

  In standard Hanabi, rank 1 has 3 copies, ranks 2-4 have 2 copies,
  and rank 5 has 1 copy.  We count how many copies have been played
  (in fireworks) or discarded, then subtract from the total.

  Args:
    state_before: State before the discard (used for fireworks).
    state_after: State after the discard (used for full discard pile).
    color: The card's colour letter.
    rank: The card's rank (1-indexed).
    player_id: Player perspective for observation.

  Returns:
    Number of copies of this card still available (in deck or hands).
  """
  total_copies = _RANK_COPIES.get(rank, 2)

  # Copies in fireworks: if the firework for this colour is >= this rank,
  # then one copy has been played.
  fireworks = _parse_fireworks_from_state(state_before, player_id)
  played = 1 if fireworks.get(color, 0) >= rank else 0

  # Copies in discard pile (after the discard action).
  after_obs = state_after.observation_string(player_id)
  card_str = f'{color}{rank}'
  discarded = 0
  for line in after_obs.split('\n'):
    if line.strip().startswith('Discards:'):
      cards = line.split(':', 1)[1].strip().split()
      discarded = cards.count(card_str)
      break

  return max(0, total_copies - played - discarded)


# =============================================================================
# Hint evaluation
# =============================================================================


def _evaluate_hint(
    state,
    hinter_id: int,
    target_offset: int,
    hint_type: str,
    hint_value: str,
) -> float:
  """Evaluate a Hint (Reveal) action by its information gain.

  Measures how much new information the hint delivers to the target
  player, with a bonus if the hint enables an immediately playable
  card.

  Information gain is measured by counting cards in the target's hand
  whose knowledge changes as a result of this hint.  Each card that
  gains new information counts as one "fact".

  Args:
    state: The pre-action state.
    hinter_id: The player giving the hint.
    target_offset: Offset to the hint target (e.g. +1 in 2-player).
    hint_type: Either 'color' or 'rank'.
    hint_value: The hinted colour letter or rank digit string.

  Returns:
    A reward in [-0.2, +0.5].
  """
  num_players = 2  # Default; works for 2-player Hanabi.
  if hasattr(state, '_game'):
    num_players = state._game.num_players()
  target_id = (hinter_id + target_offset) % num_players

  # Get the target's card knowledge before the hint.
  target_obs_before = state.observation_string(target_id)
  knowledge_before = _CARD_KNOWLEDGE_RE.findall(target_obs_before)

  # Apply the hint and get updated knowledge.
  sim = state.clone()
  # Find the matching reveal action.
  legal = sim.legal_actions(hinter_id)
  hint_action = None
  for a in legal:
    astr = sim.action_to_string(hinter_id, a)
    if hint_type == 'color' and f'color {hint_value}' in astr:
      hint_action = a
      break
    elif hint_type == 'rank' and f'rank {hint_value}' in astr:
      hint_action = a
      break

  if hint_action is None:
    # Can't find the action -- neutral reward.
    return 0.0

  sim.apply_action(hint_action)
  target_obs_after = sim.observation_string(target_id)
  knowledge_after = _CARD_KNOWLEDGE_RE.findall(target_obs_after)

  # Count new facts: cards where knowledge changed.
  new_facts = 0
  for i in range(min(len(knowledge_before), len(knowledge_after))):
    colors_before, ranks_before = knowledge_before[i]
    colors_after, ranks_after = knowledge_after[i]
    if colors_before != colors_after or ranks_before != ranks_after:
      new_facts += 1

  if new_facts == 0:
    return -0.2  # Redundant hint -- no new information.

  # Base reward from information gain (capped lower so play-exposing hints dominate).
  # 1 fact -> +0.05, 2 facts -> +0.10, 3+ facts -> +0.15
  base_reward = min(0.05 * new_facts, 0.15)

  # Check fireworks.
  fireworks = _parse_fireworks_from_state(sim, target_id)

  # Check partner's visible cards from hinter's perspective.
  hinter_obs = state.observation_string(hinter_id)
  partner_cards = [
      (m.group(1), int(m.group(2)))
      for m in _VISIBLE_CARD_RE.finditer(hinter_obs)
  ]

  # Does this hint touch an immediately playable card in the partner's hand?
  touches_playable = False
  for color, rank in partner_cards:
    needed_rank = fireworks.get(color, 0) + 1
    if rank == needed_rank:
      if hint_type == 'color' and color == hint_value:
        touches_playable = True
        break
      elif hint_type == 'rank' and str(rank) == str(hint_value):
        touches_playable = True
        break

  # Playability bonus: +0.4 if touching an immediately playable card.
  # Also check target's updated knowledge: if a card became 100% uniquely identified & playable.
  knowledge_playable = False
  for i in range(min(len(knowledge_after), 5)):
    colors_known, ranks_known = knowledge_after[i]
    if len(colors_known) == 1 and len(ranks_known) == 1:
      needed_rank = fireworks.get(colors_known, 0) + 1
      if int(ranks_known) == needed_rank:
        knowledge_playable = True
        break

  if touches_playable or knowledge_playable:
    playability_bonus = 0.40
  else:
    # If info tokens are scarce (<= 2) and this hint does NOT expose a playable card,
    # penalize wasting an info token.
    info_tokens = state.information_tokens()
    if info_tokens <= 2:
      return -0.20  # Wasted scarce info token on non-playable card.
    playability_bonus = 0.0

  return min(base_reward + playability_bonus, 0.55)


# =============================================================================
# Chained dense evaluation
# =============================================================================


def evaluate_dense_chain(
    runner,
    action_history: list[int],
    chosen_action: int,
    target_player: int,
    serialized_state: Optional[str] = None,
    horizon: int = 4,
    discount: float = 0.9,
    llm_partner_response: bool = False,
) -> float:
  """Evaluate a chosen action plus a short heuristic continuation.

  Computes the dense reward for ``chosen_action``, then continues
  the game for ``horizon`` more turns using the heuristic player,
  accumulating discounted dense rewards for each subsequent action.

  This addresses the myopic-play concern: the agent gets credit not
  just for the immediate action quality but also for how well the
  game state it creates supports good subsequent play.

  The total reward is::

    r_0 + gamma * r_1 + gamma^2 * r_2 + ... + gamma^h * r_h

  where r_0 is the dense reward for ``chosen_action`` and r_1..r_h
  are the dense rewards for the heuristic player's subsequent moves.

  When ``llm_partner_response`` is True, the first continuation
  turn (the partner's response to the chosen action) uses the
  frozen LLM policy instead of SafePlayPlayer.  This captures
  whether the partner can actually exploit hints or play setups.

  Args:
    runner: The ``GRPORunner`` instance (for environment and config).
    action_history: Action history leading to the current state.
    chosen_action: The action to evaluate.
    target_player: The player whose perspective we're evaluating from.
    serialized_state: Serialized state string for restoration.
    horizon: Number of additional turns to simulate after the chosen
        action. Each turn's reward is discounted by ``discount``.
    discount: Discount factor gamma for future action rewards.
    llm_partner_response: If True, use frozen LLM policy for the first
        continuation turn (the partner's immediate response).

  Returns:
    The total discounted dense reward.
  """
  # Restore the game state.
  if serialized_state is not None:
    _, state = deserialize_game_and_state(serialized_state)
    runner._env.set_state(state)
  else:
    runner._env.reset()
    state = runner._env._state
    for a in action_history:
      if state.is_terminal():
        break
      state.apply_action(a)

  state = runner._env._state
  if state.is_terminal():
    return 0.0

  # Compute dense reward for the chosen action.
  total_reward = evaluate_action_quality(state, chosen_action, target_player)

  # Apply the chosen action.
  state.apply_action(chosen_action)

  # Continue with heuristic player for `horizon` turns.
  bot_type = getattr(runner._config, 'bot_type', 'belief_lookahead')
  heuristic = None
  game = getattr(runner._env, 'game', None)
  if bot_type == 'belief_lookahead':
    try:
      from env.hanabi.belief_expert import SafeBeliefLookaheadPlayer  # pylint: disable=g-import-not-at-top
      heuristic = SafeBeliefLookaheadPlayer(game, n_worlds=1, seed=42)
    except Exception:
      heuristic = None
  if heuristic is None:
    try:
      from env.hanabi.heuristic_player import SafePlayPlayer  # pylint: disable=g-import-not-at-top
      heuristic = SafePlayPlayer(seed=42)
    except ImportError:
      # No heuristic available -- return just the immediate reward.
      return total_reward
  gamma = discount
  first_continuation = True
  for _ in range(horizon):
    if state.is_terminal():
      break
    current_player = state.current_player()
    legal = state.legal_actions(current_player)
    if not legal:
      break

    # For the first continuation turn, optionally use the LLM
    # (the partner's response to our action).
    if first_continuation and llm_partner_response:
      first_continuation = False
      from learn.grpo_sampled import _sample_llm_partner_action  # pylint: disable=g-import-not-at-top
      h_action = _sample_llm_partner_action(runner, state)
      if h_action is None:
        h_action = int(np.random.choice(legal))
    else:
      first_continuation = False
      # Heuristic selects the next action.
      h_action = heuristic.select_action(state, current_player, game)
      if h_action is None:
        h_action = int(np.random.choice(legal))

    # Compute dense reward for this continuation action.
    step_reward = evaluate_action_quality(
        state, h_action, current_player
    )
    total_reward += gamma * step_reward
    gamma *= discount

    state.apply_action(h_action)

  return total_reward
