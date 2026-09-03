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
    to -0.7 (discarding a currently playable card).
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

import re
from typing import Optional

from absl import logging
import numpy as np


# -- Constants ----------------------------------------------------------------

_COLOR_CHARS = ('R', 'Y', 'G', 'W', 'B')
_PLAY_RE = re.compile(r'\(Play (\d+)\)')
_DISCARD_RE = re.compile(r'\(Discard (\d+)\)')
_REVEAL_COLOR_RE = re.compile(
    r'\(Reveal player \+(\d+) color ([RYGWB])\)'
)
_REVEAL_RANK_RE = re.compile(
    r'\(Reveal player \+(\d+) rank (\d+)\)'
)
_FIREWORKS_RE = re.compile(r'Fireworks:\s*((?:[RYGWB]\d\s*)+)')
_CARD_KNOWLEDGE_RE = re.compile(
    r'XX\s*\|\|\s*(?:[A-Z0-9]+[|])?([RYGWB]+)[|]?([1-5]+)'
)

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
    return _evaluate_play(state, action_id)

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
# Play evaluation
# =============================================================================


def _evaluate_play(state, action_id: int) -> float:
  """Evaluate a Play action by checking the state transition.

  Clones the state, applies the action, and checks whether the score
  increased (successful play) or a life token was lost (failed play).

  Args:
    state: The pre-action state.
    action_id: The Play action UID.

  Returns:
    +1.0 for a valid play, +1.5 if it completes a colour, -1.0 for
    an invalid play.
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
    return -1.0  # Invalid play -- lost a life token.
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

  Uses the game state to determine:
    - Is the card already played (dead)? -> safe discard.
    - Is it the last copy of a still-needed card? -> critical loss.
    - Is it currently playable? -> wasted opportunity.
    - Otherwise: routine discard.

  Args:
    state: The pre-action state.
    action_id: The Discard action UID.
    player_id: The acting player.
    card_pos: The card's position in hand (0-indexed).

  Returns:
    A reward in [-0.7, +0.3].
  """
  # We need to know the actual card being discarded.
  # The acting player can't see their own cards, but the *state* knows.
  # We clone and inspect the discard pile to identify the card.
  score_before = state.score()
  fireworks = _parse_fireworks_from_state(state, player_id)

  sim = state.clone()
  sim.apply_action(action_id)

  # Identify the discarded card by diffing the observation.
  # After a discard, the card appears in the discard pile.
  discarded_card = _identify_discarded_card(state, sim, player_id)
  if discarded_card is None:
    # Can't determine the card -- give a small positive reward
    # (discarding regains an info token, which has some value).
    return 0.05

  card_color, card_rank = discarded_card

  # Case 1: Card is already played (rank <= firework height).
  fw_height = fireworks.get(card_color, 0)
  if card_rank <= fw_height:
    return 0.3  # Dead card -- excellent discard.

  # Case 2: Card is currently playable (rank == firework height + 1).
  if card_rank == fw_height + 1:
    return -0.7  # Discarding a playable card.

  # Case 3: Card is the last copy of a still-needed card.
  remaining = _count_remaining_copies(
      state, sim, card_color, card_rank, player_id
  )
  if remaining == 0:
    # This was the last copy and we just discarded it.
    return -0.5  # Permanently reduces maximum achievable score.

  # Case 4: Card has other copies remaining.
  # Mildly positive -- regains info token, not critical.
  return 0.1


def _parse_fireworks_from_state(state, player_id: int) -> dict[str, int]:
  """Extract firework heights from the state's observation string."""
  obs = state.observation_string(player_id)
  fireworks: dict[str, int] = {c: 0 for c in _COLOR_CHARS}
  match = _FIREWORKS_RE.search(obs)
  if match:
    for token in match.group(1).strip().split():
      if len(token) >= 2:
        fireworks[token[0]] = int(token[1:])
  return fireworks


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

  # Base reward from information gain.
  # 1 fact -> +0.1, 2 facts -> +0.2, 3+ facts -> +0.3
  base_reward = min(0.1 * new_facts, 0.3)

  # Bonus: does this hint enable an immediately playable card?
  # Check if any of the target's cards are now identifiable as playable.
  fireworks = _parse_fireworks_from_state(sim, target_id)
  playability_bonus = 0.0
  for i in range(min(len(knowledge_after), 5)):
    colors_known, ranks_known = knowledge_after[i]
    if len(colors_known) == 1 and len(ranks_known) == 1:
      needed_rank = fireworks.get(colors_known, 0) + 1
      if int(ranks_known) == needed_rank:
        # This card is now known to be playable!
        playability_bonus = 0.2
        break

  return min(base_reward + playability_bonus, 0.5)


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

  Args:
    runner: The ``GRPORunner`` instance (for environment and config).
    action_history: Action history leading to the current state.
    chosen_action: The action to evaluate.
    target_player: The player whose perspective we're evaluating from.
    serialized_state: Serialized state string for restoration.
    horizon: Number of additional turns to simulate after the chosen
        action. Each turn's reward is discounted by ``discount``.
    discount: Discount factor gamma for future action rewards.

  Returns:
    The total discounted dense reward.
  """
  # Restore the game state.
  if serialized_state is not None:
    try:
      from env.hanabi.hanabi_env import deserialize_game_and_state  # pylint: disable=g-import-not-at-top
      _, state = deserialize_game_and_state(serialized_state)
    except (ImportError, Exception):
      from learn.grpo_sampled import _deserialize_game_and_state  # pylint: disable=g-import-not-at-top
      _, state = _deserialize_game_and_state(serialized_state)
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
  try:
    from env.hanabi.heuristic_player import SafePlayPlayer  # pylint: disable=g-import-not-at-top
    heuristic = SafePlayPlayer()
  except ImportError:
    # No heuristic available -- return just the immediate reward.
    return total_reward

  game = getattr(runner._env, 'game', None)
  gamma = discount
  for _ in range(horizon):
    if state.is_terminal():
      break
    current_player = state.current_player()
    legal = state.legal_actions(current_player)
    if not legal:
      break

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
