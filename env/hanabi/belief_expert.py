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

"""Belief-state lookahead expert player for Hanabi.

Evaluates candidate actions using belief-consistent determinization and
heuristic rollouts with SafePlayPlayer, while enforcing safe play constraints.
"""

from __future__ import annotations

import random
from typing import Any, Optional

import numpy as np

from env.hanabi import determinize
from env.hanabi import heuristic_player


def _rollout_to_terminal(state: Any, players: list[Any], max_turns: int = 200) -> int:
  """Rollout game to terminal using heuristic players."""
  turns = 0
  while not state.is_terminal() and turns < max_turns:
    cur = state.current_player()
    if cur < 0:
      break
    action = players[cur].select_action(state, cur)
    state.apply_action(action)
    turns += 1
  return state.score()


class SafeBeliefLookaheadPlayer:
  """Belief lookahead expert player for Hanabi.

  Priorities:
    1. If a card is known playable from card knowledge, play it immediately
       (100% safe, avoids bomb-outs).
    2. For remaining turns (hints and discards), sample N determinized worlds
       consistent with own card knowledge and observation, roll each candidate
       action forward with SafePlayPlayer, and pick the action that maximizes
       expected final score.
  """

  def __init__(
      self,
      game: Any,
      determinizer: Optional[determinize.Determinizer] = None,
      n_worlds: int = 3,
      seed: Optional[int] = None,
      new_cards_at_index_zero: bool = False,
  ) -> None:
    self._game = game
    self._determinizer = determinizer or determinize.Determinizer(game)
    self._n_worlds = n_worlds
    self._seed = seed
    self._rng = random.Random(seed)
    self._bot = heuristic_player.SafePlayPlayer(
        seed=seed, new_cards_at_index_zero=new_cards_at_index_zero
    )
    self._rollout_players = [
        heuristic_player.SafePlayPlayer(
            seed=seed, new_cards_at_index_zero=new_cards_at_index_zero
        ),
        heuristic_player.SafePlayPlayer(
            seed=(seed + 1000) if seed is not None else None,
            new_cards_at_index_zero=new_cards_at_index_zero,
        ),
    ]

  def select_action(
      self,
      state: Any,
      player_id: int,
      game: Optional[Any] = None,
  ) -> int:
    """Select the best action via belief lookahead with safe play constraint."""
    del game
    legal_actions = state.legal_actions(player_id)
    if len(legal_actions) == 1:
      return legal_actions[0]

    obs_string = state.observation_string(player_id)
    fireworks = self._bot._parse_fireworks(obs_string)
    card_knowledge = self._bot._parse_card_knowledge(obs_string)
    play_actions, discard_actions, hint_actions = self._bot._classify_actions(
        state, player_id, legal_actions
    )

    # 1. Known-safe play: 100% success rate
    playable_action = self._bot._find_known_playable(
        card_knowledge, fireworks, play_actions
    )
    if playable_action is not None:
      return playable_action

    # 2. Restrict candidate actions to legal hints and legal discards.
    # We deliberately do NOT gamble on blind plays of unhinted cards.
    candidate_actions = list(hint_actions)
    if state.information_tokens() < 8 and discard_actions:
      candidate_actions.extend(discard_actions)
    elif not candidate_actions:
      candidate_actions = list(discard_actions)

    if not candidate_actions:
      candidate_actions = legal_actions

    if len(candidate_actions) == 1:
      return candidate_actions[0]

    # Sample N determinized worlds
    worlds = []
    for _ in range(self._n_worlds * 3):
      w = self._determinizer.determinize(state, player_id, self._rng)
      if w is not None:
        worlds.append(w)
      if len(worlds) >= self._n_worlds:
        break

    if not worlds:
      return self._bot.select_action(state, player_id)

    best_action = candidate_actions[0]
    best_q = -1e9

    for action in candidate_actions:
      total_score = 0.0
      for w in worlds:
        sim = w.clone()
        sim.apply_action(action)
        if sim.is_terminal():
          total_score += sim.score()
        else:
          total_score += _rollout_to_terminal(sim, self._rollout_players)
      q = total_score / len(worlds)
      if q > best_q:
        best_q = q
        best_action = action

    return best_action
