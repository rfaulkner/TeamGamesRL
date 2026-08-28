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

"""Game-tree enumeration and oracle strategy computation for GRPO.

This module contains the functions that walk an OpenSpiel game tree to
construct GRPO groups, compute oracle strategies, and simulate game
completions.  These are the pieces most likely to require adaptation
when extending TeamGamesRL to new, larger games.

All public functions accept a ``runner`` parameter — the ``GRPORunner``
instance — which provides access to the environment, renderers, agents,
backend, and configuration.
"""

from itertools import product as iter_product

from absl import logging
import numpy as np
import pyspiel
import torch


# ═══════════════════════════════════════════════════════════════════════
# Group deduplication (shared helper)
# ═══════════════════════════════════════════════════════════════════════


def deduplicate_groups(
    groups: list[dict[str, object]],
) -> list[dict[str, object]]:
  """Remove groups with identical (player_id, prompt, rewards).

  Multiple chance-outcome paths can lead to the same observable state
  and reward structure.  Deduplication avoids redundant gradient updates.

  Args:
    groups: Raw list of GRPO group dicts.

  Returns:
    Deduplicated list preserving original order.
  """
  seen: set[str] = set()
  unique: list[dict[str, object]] = []
  for g in groups:
    key = str((g['player_id'], g['prompt'], tuple(g['rewards'])))
    if key not in seen:
      seen.add(key)
      unique.append(g)
  return unique


# ═══════════════════════════════════════════════════════════════════════
# Pure helpers (no runner / model dependency)
# ═══════════════════════════════════════════════════════════════════════


def max_reward_over_partners(
    state: pyspiel.State,
    target_player: int,
) -> float:
  """Compute the max reward over all partner actions from ``state``.

  Enumerates all legal actions for the current player at ``state`` and
  recursively finds the maximum achievable reward for ``target_player``.
  This gives an optimistic estimate: "what's the best reward if my
  partner cooperated perfectly?"

  At chance nodes, takes the max over all outcomes (fully optimistic).
  At the target player's own decision nodes (if any remain), also
  takes the max.

  Args:
    state: The game state to evaluate (typically after the target
      player has already acted).
    target_player: The player whose reward to maximize.

  Returns:
    The maximum possible terminal reward for ``target_player``.
  """
  if state.is_terminal():
    rewards = state.rewards()
    return float(rewards[target_player]) if rewards else 0.0

  if state.is_chance_node():
    best = float('-inf')
    for chance_action, _ in state.chance_outcomes():
      r = max_reward_over_partners(state.child(chance_action), target_player)
      best = max(best, r)
    return best

  # Player decision node — enumerate all actions and take the max.
  current_player = state.current_player()
  best = float('-inf')
  for action in state.legal_actions(current_player):
    r = max_reward_over_partners(state.child(action), target_player)
    best = max(best, r)
  return best


def compute_oracle_p0_strategy(runner) -> dict[int, int]:
  """Compute the optimal P0 signaling strategy for the game.

  Solves for the Stackelberg-optimal P0 pure strategy by brute-forcing
  all possible mappings from P0's card to P0's action, and for each
  one computing P1's best response **given P1's information
  constraint** (P1 observes P0's action and P1's own card, but NOT
  P0's card).

  This is necessary because ``max_reward_over_partners`` assumes P1
  can see the full state, which makes every P0 action look equally
  good.  The real oracle must account for the fact that P1 must play
  the same action for all P0 cards that map to the same P0 action.

  Args:
    runner: The ``GRPORunner`` instance.

  Returns:
    Dict mapping P0's card (chance-action int) to the optimal P0
    action (int).
  """
  game = runner._env.game

  # ── Step 1: Walk the full game tree to collect all terminal payoffs ──
  payoffs: dict[tuple[int, ...], float] = {}

  def _walk_all(state, deal, actions):
    if state.is_terminal():
      rewards = state.rewards()
      team_r = float(np.mean(rewards)) if rewards else 0.0
      payoffs[tuple(deal + actions)] = team_r
      return
    if state.is_chance_node():
      for action, _ in state.chance_outcomes():
        _walk_all(state.child(action), deal + [action], actions)
      return
    for action in state.legal_actions(state.current_player()):
      _walk_all(state.child(action), deal, actions + [action])

  _walk_all(game.new_initial_state(), [], [])

  # ── Step 2: Extract game dimensions ──
  p0_cards = sorted(set(k[0] for k in payoffs))
  p1_cards = sorted(set(k[1] for k in payoffs))
  p0_actions = sorted(set(k[2] for k in payoffs))
  p1_actions = sorted(set(k[3] for k in payoffs))

  logging.info(
      'Computing optimal P0 oracle: %d P0 cards × %d P0 actions '
      '= %d strategies to check.',
      len(p0_cards), len(p0_actions), len(p0_actions) ** len(p0_cards),
  )

  # ── Step 3: Brute-force all P0 pure strategies ──
  best_strategy: dict[int, int] = {}
  best_expected_reward = float('-inf')
  best_p1_response: dict[tuple[int, int], int] = {}

  for p0_strategy_actions in iter_product(
      p0_actions, repeat=len(p0_cards)
  ):
    p0_map = dict(zip(p0_cards, p0_strategy_actions))

    # For this P0 strategy, compute P1's best response.
    p1_response: dict[tuple[int, int], int] = {}
    for p1_card in p1_cards:
      for p0_action in set(p0_map.values()):
        p0_cards_for_action = [
            c for c in p0_cards if p0_map[c] == p0_action
        ]
        best_p1_a = p1_actions[0]
        best_p1_r = float('-inf')
        for p1_action in p1_actions:
          expected = float(np.mean([
              payoffs.get((p0c, p1_card, p0_action, p1_action), 0.0)
              for p0c in p0_cards_for_action
          ]))
          if expected > best_p1_r:
            best_p1_r = expected
            best_p1_a = p1_action
        p1_response[(p1_card, p0_action)] = best_p1_a

    # Compute expected reward.
    total = 0.0
    count = 0
    for p0_card in p0_cards:
      for p1_card in p1_cards:
        p0_action = p0_map[p0_card]
        p1_action = p1_response[(p1_card, p0_action)]
        total += payoffs.get(
            (p0_card, p1_card, p0_action, p1_action), 0.0
        )
        count += 1

    expected_reward = total / count if count else 0.0
    if expected_reward > best_expected_reward:
      best_expected_reward = expected_reward
      best_strategy = dict(p0_map)
      best_p1_response = dict(p1_response)

  # ── Log ──
  logging.info(
      'Optimal P0 oracle strategy (expected reward=%.2f):',
      best_expected_reward,
  )
  for card in sorted(best_strategy):
    logging.info('  P0 card=%d → action=%d', card, best_strategy[card])
  logging.info('P1 best response to oracle P0:')
  for (p1_card, p0_action), p1_action in sorted(
      best_p1_response.items()
  ):
    logging.info(
        '  P1 card=%d, saw P0 action=%d → P1 action=%d',
        p1_card, p0_action, p1_action,
    )
  return best_strategy


# ═══════════════════════════════════════════════════════════════════════
# Play-out / simulation helpers
# ═══════════════════════════════════════════════════════════════════════


def play_out_for_reward(
    runner,
    state: pyspiel.State,
    target_player: int,
) -> float:
  """Play out a game from ``state`` to completion, returning the reward.

  At each remaining decision point, uses the LLM policy (with
  ``torch.no_grad()``) to select actions.

  Args:
    runner: The ``GRPORunner`` instance.
    state: The game state to play from (modified in place).
    target_player: The player whose reward to return.

  Returns:
    The terminal reward for ``target_player``.
  """
  game = runner._env.game

  while not state.is_terminal():
    if state.is_chance_node():
      outcomes = state.chance_outcomes()
      actions, probs = zip(*outcomes)
      state.apply_action(int(np.random.choice(actions, p=probs)))
      continue

    current_player = state.current_player()
    state_text = runner._renderers[current_player].render_state(
        state, current_player, game
    )
    legal_actions_with_desc = runner._renderers[
        current_player
    ].render_legal_actions(state, current_player, game)
    legal_ids = [a for a, _ in legal_actions_with_desc]
    action_descs = [d for _, d in legal_actions_with_desc]

    prompt = runner._agents[current_player]._build_prompt(  # pylint: disable=protected-access
        state_text, legal_ids, action_descs
    )
    with torch.no_grad():
      response, _ = runner._backend.generate_with_logprobs(
          prompt,
          temperature=runner._current_temperature,
          max_tokens=runner._config.max_completion_length,
      )

    action_id = runner._renderers[current_player].parse_action(
        response, legal_actions_with_desc
    )
    if action_id is None:
      action_id = int(np.random.choice(legal_ids))
    state.apply_action(action_id)

  rewards = state.rewards()
  return float(rewards[target_player]) if rewards is not None else 0.0


def simulate_with_adapter(
    runner,
    state: pyspiel.State,
    target_player: int,
    other_adapter: str,
) -> float:
  """Simulate a game to completion using a specific adapter for the other player.

  Args:
    runner: The ``GRPORunner`` instance.
    state: The game state to play from (cloned internally).
    target_player: The player whose reward to return.
    other_adapter: Name of the adapter to use for the other player.

  Returns:
    The terminal reward for ``target_player``.
  """
  game = runner._env.game
  state = state.clone()

  while not state.is_terminal():
    if state.is_chance_node():
      outcomes = state.chance_outcomes()
      actions, probs = zip(*outcomes)
      state.apply_action(int(np.random.choice(actions, p=probs)))
      continue

    current_player = state.current_player()
    prev_adapter = runner._backend.get_active_adapter()
    runner._backend.set_active_adapter(
        other_adapter if current_player != target_player
        else f'player_{target_player}'
    )

    state_text = runner._renderers[current_player].render_state(
        state, current_player, game
    )
    legal_actions_with_desc = runner._renderers[
        current_player
    ].render_legal_actions(state, current_player, game)
    legal_ids = [a for a, _ in legal_actions_with_desc]
    action_descs = [d for _, d in legal_actions_with_desc]
    prompt = runner._agents[current_player]._build_prompt(  # pylint: disable=protected-access
        state_text, legal_ids, action_descs
    )

    with torch.no_grad():
      response, _ = runner._backend.generate_with_logprobs(
          prompt,
          temperature=0.01,  # Near-greedy.
          max_tokens=runner._config.max_completion_length,
      )

    runner._backend.set_active_adapter(prev_adapter)

    action_id = runner._renderers[current_player].parse_action(
        response, legal_actions_with_desc
    )
    if action_id is None:
      action_id = int(np.random.choice(legal_ids))
    state.apply_action(action_id)

  rewards = state.rewards()
  return float(rewards[target_player]) if rewards else 0.0


# ═══════════════════════════════════════════════════════════════════════
# Full game-tree enumeration
# ═══════════════════════════════════════════════════════════════════════


def enumerate_grpo_groups(
    runner,
    optimistic_alpha: float = 0.0,
) -> list[dict[str, object]]:
  """Enumerate all GRPO groups by walking the game tree.

  A *group* is a set of (prompt, action, reward) tuples where all game
  variables — chance outcomes and other players' actions — are held fixed
  and only the *target player's action* varies.  This gives the cleanest
  possible GRPO advantage signal: zero variance, no cross-context noise.

  Args:
    runner: The ``GRPORunner`` instance.
    optimistic_alpha: Blending weight for optimistic rewards.

  Returns:
    Deduplicated list of group dicts, each containing ``player_id``,
    ``prompt``, ``actions``, ``action_texts``, ``rewards``, ``context``.
  """
  game = runner._env.game
  groups: list[dict[str, object]] = []

  def _walk(state, context_parts, target_player):
    if state.is_terminal():
      return

    if state.is_chance_node():
      for chance_action, _ in state.chance_outcomes():
        action_str = state.action_to_string(
            pyspiel.PlayerId.CHANCE, chance_action
        )
        _walk(
            state.child(chance_action),
            context_parts + [f'chance:{action_str}'],
            target_player,
        )
      return

    current_player = state.current_player()

    if target_player is None or current_player == target_player:
      # Decision point to expand into a GRPO group.
      state_text = runner._renderers[current_player].render_state(
          state, current_player, game
      )
      legal_actions_with_desc = runner._renderers[
          current_player
      ].render_legal_actions(state, current_player, game)
      legal_ids = [a for a, _ in legal_actions_with_desc]
      action_descs = [d for _, d in legal_actions_with_desc]

      prompt = runner._agents[current_player]._build_prompt(  # pylint: disable=protected-access
          state_text, legal_ids, action_descs
      )

      # Compute reward for each action.
      action_rewards: list[float] = []
      action_texts: list[str] = []
      for action_id in legal_ids:
        child = state.child(action_id)
        max_r = None

        if optimistic_alpha > 0 and not child.is_terminal():
          max_r = max_reward_over_partners(child, current_player)

        sim_reward = play_out_for_reward(
            runner, state.child(action_id), current_player
        )

        if optimistic_alpha > 0 and max_r is not None:
          reward = optimistic_alpha * max_r + (1.0 - optimistic_alpha) * sim_reward
        else:
          reward = sim_reward

        action_rewards.append(reward)
        action_texts.append(action_descs[legal_ids.index(action_id)])

      groups.append({
          'player_id': current_player,
          'prompt': prompt,
          'actions': legal_ids,
          'action_texts': action_texts,
          'rewards': action_rewards,
          'context': ', '.join(context_parts),
      })

      # Recurse for downstream players.
      for action_id in legal_ids:
        action_str = state.action_to_string(current_player, action_id)
        _walk(
            state.child(action_id),
            context_parts + [f'p{current_player}:{action_str}'],
            None,
        )

    else:
      # Other player's decision — use LLM to pick a single action.
      state_text = runner._renderers[current_player].render_state(
          state, current_player, game
      )
      legal_actions_with_desc = runner._renderers[
          current_player
      ].render_legal_actions(state, current_player, game)
      legal_ids = [a for a, _ in legal_actions_with_desc]
      action_descs = [d for _, d in legal_actions_with_desc]

      prompt = runner._agents[current_player]._build_prompt(  # pylint: disable=protected-access
          state_text, legal_ids, action_descs
      )
      with torch.no_grad():
        response, _ = runner._backend.generate_with_logprobs(
            prompt,
            temperature=runner._current_temperature,
            max_tokens=runner._config.max_completion_length,
        )

      action_id = runner._renderers[current_player].parse_action(
          response, legal_actions_with_desc
      )
      if action_id is None:
        action_id = int(np.random.choice(legal_ids))

      action_str = state.action_to_string(current_player, action_id)
      _walk(
          state.child(action_id),
          context_parts + [f'p{current_player}:{action_str}'],
          target_player,
      )

  _walk(game.new_initial_state(), [], None)

  unique = deduplicate_groups(groups)
  logging.info(
      'Enumerated %d GRPO groups (%d before dedup).',
      len(unique), len(groups),
  )
  return unique


def enumerate_single_player_groups(
    runner,
    target_player_id: int,
    other_player_mode: str = 'oracle',
    oracle_strategy: dict[int, int] | None = None,
) -> list[dict[str, object]]:
  """Enumerate GRPO groups for a single player.

  Unlike ``enumerate_grpo_groups`` which creates groups for *both*
  players, this method creates groups only for ``target_player_id``.

  Args:
    runner: The ``GRPORunner`` instance.
    target_player_id: The player whose groups to enumerate (0 or 1).
    other_player_mode: ``'oracle'`` or ``'simulate'``.
    oracle_strategy: Pre-computed oracle P0 strategy (required when
      ``other_player_mode='oracle'``).

  Returns:
    Deduplicated list of group dicts.
  """
  game = runner._env.game
  groups: list[dict[str, object]] = []
  other_player_id = 1 - target_player_id

  def _walk(state, context_parts):
    if state.is_terminal():
      return

    if state.is_chance_node():
      for chance_action, _ in state.chance_outcomes():
        action_str = state.action_to_string(
            pyspiel.PlayerId.CHANCE, chance_action
        )
        _walk(state.child(chance_action),
              context_parts + [f'chance:{action_str}'])
      return

    current_player = state.current_player()

    if current_player == target_player_id:
      # Expand into a GRPO group for the target player.
      state_text = runner._renderers[current_player].render_state(
          state, current_player, game
      )
      legal_actions_with_desc = runner._renderers[
          current_player
      ].render_legal_actions(state, current_player, game)
      legal_ids = [a for a, _ in legal_actions_with_desc]
      action_descs = [d for _, d in legal_actions_with_desc]

      prompt = runner._agents[current_player]._build_prompt(  # pylint: disable=protected-access
          state_text, legal_ids, action_descs
      )

      action_rewards: list[float] = []
      action_texts: list[str] = []
      for action_id in legal_ids:
        child = state.child(action_id)
        if child.is_terminal():
          rewards = child.rewards()
          reward = float(rewards[target_player_id]) if rewards else 0.0
        elif other_player_mode == 'oracle':
          reward = max_reward_over_partners(child, target_player_id)
        else:
          reward = simulate_with_adapter(
              runner, child, target_player_id,
              f'player_{other_player_id}',
          )
        action_rewards.append(reward)
        action_texts.append(action_descs[legal_ids.index(action_id)])

      groups.append({
          'player_id': current_player,
          'prompt': prompt,
          'actions': legal_ids,
          'action_texts': action_texts,
          'rewards': action_rewards,
          'context': ', '.join(context_parts),
      })

      for action_id in legal_ids:
        action_str = state.action_to_string(current_player, action_id)
        _walk(
            state.child(action_id),
            context_parts + [f'p{current_player}:{action_str}'],
        )

    else:
      # Other player's decision node.
      if other_player_mode == 'oracle':
        _walk_oracle_other(
            runner, state, context_parts, current_player,
            oracle_strategy, game, _walk,
        )
      else:
        _walk_simulate_other(
            runner, state, context_parts, current_player, game, _walk,
        )

  _walk(game.new_initial_state(), [])

  unique = deduplicate_groups(groups)
  logging.info(
      'Enumerated %d %s groups for P%d (%d before dedup).',
      len(unique), other_player_mode, target_player_id, len(groups),
  )
  return unique


def _walk_oracle_other(runner, state, context_parts, current_player,
                       oracle_strategy, game, walk_fn):
  """Handle the other player's node in oracle mode."""
  if oracle_strategy is None:
    # Fallback if oracle strategy wasn't computed.
    for action_id in state.legal_actions(current_player):
      action_str = state.action_to_string(current_player, action_id)
      walk_fn(
          state.child(action_id),
          context_parts + [f'p{current_player}:{action_str}'],
      )
    return

  history = state.history()
  other_card = history[current_player]
  oracle_action = oracle_strategy.get(other_card)

  if (oracle_action is not None
      and oracle_action in state.legal_actions(current_player)):
    action_str = state.action_to_string(current_player, oracle_action)
    walk_fn(
        state.child(oracle_action),
        context_parts + [f'p{current_player}:{action_str}'],
    )
  else:
    logging.warning(
        'Oracle strategy missing for P%d card=%d, '
        'falling back to all actions.',
        current_player, other_card,
    )
    for action_id in state.legal_actions(current_player):
      action_str = state.action_to_string(current_player, action_id)
      walk_fn(
          state.child(action_id),
          context_parts + [f'p{current_player}:{action_str}'],
      )


def _walk_simulate_other(runner, state, context_parts, current_player,
                         game, walk_fn):
  """Handle the other player's node by simulating with their adapter."""
  prev_adapter = runner._backend.get_active_adapter()
  runner._backend.set_active_adapter(f'player_{current_player}')

  state_text = runner._renderers[current_player].render_state(
      state, current_player, game
  )
  legal_actions_with_desc = runner._renderers[
      current_player
  ].render_legal_actions(state, current_player, game)
  legal_ids = [a for a, _ in legal_actions_with_desc]
  action_descs = [d for _, d in legal_actions_with_desc]
  prompt = runner._agents[current_player]._build_prompt(  # pylint: disable=protected-access
      state_text, legal_ids, action_descs
  )

  with torch.no_grad():
    response, _ = runner._backend.generate_with_logprobs(
        prompt,
        temperature=0.01,
        max_tokens=runner._config.max_completion_length,
    )

  runner._backend.set_active_adapter(prev_adapter)

  action_id = runner._renderers[current_player].parse_action(
      response, legal_actions_with_desc
  )
  if action_id is None:
    action_id = int(np.random.choice(legal_ids))

  action_str = state.action_to_string(current_player, action_id)
  walk_fn(
      state.child(action_id),
      context_parts + [f'p{current_player}:{action_str}'],
  )
