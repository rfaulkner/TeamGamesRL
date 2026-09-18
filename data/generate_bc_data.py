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

"""Generates Supervised Fine-Tuning (BC) data for Hanabi using Belief Expert."""

import argparse
import json
import pathlib
import random
import sys
import time

import types
_stub = types.ModuleType('pyspiel')
_stub.State = object
_stub.Game = object
sys.modules.setdefault('pyspiel', _stub)

from env import state_renderers
from env.hanabi import belief_expert
from env.hanabi import determinize
from env.hanabi import hanabi_env

_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert game-playing AI agent. You are playing the game: {game_name}.

{game_description}

RULES:
- You must select exactly one action from the list of legal actions provided.
- Respond with ONLY the action description — nothing else.
- Do NOT add explanations, reasoning, commentary, or newlines.
- Copy the action text exactly as shown in the legal actions list.
- Think strategically to maximize your chance of winning.

You are Player {player_id}.
"""

_SYSTEM_PROMPT_REASONING_TEMPLATE = """\
You are an expert game-playing AI agent. You are playing the game: {game_name}.

{game_description}

RULES:
- You must select exactly one action from the list of legal actions provided.
- First, analyze the current situation step-by-step inside <think>...</think>. Consider:
  1. Fireworks status and remaining life / info tokens.
  2. Cards in your hand: identify which cards are likely playable based on partner clues (matching active firework stacks), safe to discard, or uncertain.
  3. Playable or critical cards in your partner's hand that need hints.
  4. Which action (Play, Discard, or Hint) creates the highest expected game value (remember: advancing score requires playing cards; taking calculated risks on hinted cards is necessary).
- After </think>, output the chosen action on a new line, matching the legal actions list.

You are Player {player_id}.
"""

_USER_PROMPT_TEMPLATE = """\
Current game state:
{state_text}

Legal actions:
{actions_text}

Action:"""


def build_system_prompt(game, player_id: int, reasoning: bool = False) -> str:
  game_type = game.get_type()
  game_description = (
      f'Game type: {game_type.short_name}\n'
      f'Number of players: {game.num_players()}\n'
      f'Number of distinct actions: {game.num_distinct_actions()}'
  )
  template = _SYSTEM_PROMPT_REASONING_TEMPLATE if reasoning else _SYSTEM_PROMPT_TEMPLATE
  return template.format(
      game_name=game_type.short_name,
      game_description=game_description,
      player_id=player_id,
  )


def generate_cot_reasoning(state, player_id: int, target_desc: str, bot) -> str:
  """Generates a concise, structured reasoning block for the chosen action."""
  obs_string = state.observation_string(player_id)
  fireworks = bot._parse_fireworks(obs_string)
  info_tokens = state.information_tokens()
  lives = state.life_tokens()

  fw_str = ', '.join(f'{c}:{h}' for c, h in sorted(fireworks.items()))

  reasons = [f'Fireworks: {fw_str} | Info: {info_tokens}/8 | Lives: {lives}/3.']

  action_lower = target_desc.lower()
  if action_lower.startswith('hint') or 'reveal' in action_lower or 'hint ' in action_lower:
    reasons.append(
        f'No confident playable card in hand, but info tokens ({info_tokens}) are available.'
    )
    reasons.append(
        'Providing this clue guides partner toward a safe play or protects a critical card.'
    )
  elif action_lower.startswith('discard'):
    reasons.append(
        f'No clear play available and info tokens ({info_tokens}) can be replenished.'
    )
    reasons.append(
        'Discarding an unhinted or dead card regains 1 info token safely.'
    )
  elif action_lower.startswith('play'):
    reasons.append(
        'Card clues indicate this card is likely playable on the fireworks stacks.'
    )
    reasons.append('Playing advances team score towards completing the fireworks.')
  else:
    reasons.append('Evaluating legal options to maximize expected team score.')

  reasons.append(f'Best action: {target_desc}.')
  think_body = '\n'.join(f'- {r}' for r in reasons)
  return f'<think>\n{think_body}\n</think>\n{target_desc}'


def build_full_prompt(system_prompt: str, state_text: str, action_descriptions: list[str]) -> str:
  actions_lines = [f'  - {desc}' for desc in action_descriptions]
  actions_text = '\n'.join(actions_lines)
  user_prompt = _USER_PROMPT_TEMPLATE.format(
      state_text=state_text,
      actions_text=actions_text,
  )
  return f'{system_prompt}\n{user_prompt}'


def main():
  parser = argparse.ArgumentParser(description='Generate BC data for Hanabi.')
  parser.add_argument('--num_games', type=int, default=50, help='Number of self-play games to run.')
  parser.add_argument('--n_worlds', type=int, default=3, help='Determinized worlds per decision.')
  parser.add_argument('--max_history_turns', type=int, default=20, help='History turns in renderer.')
  parser.add_argument('--output_dir', type=str, default='data/bc_hanabi', help='Directory to write JSONL.')
  parser.add_argument('--train_split', type=float, default=0.9, help='Fraction of data for training.')
  parser.add_argument('--seed', type=int, default=42, help='Base random seed.')
  parser.add_argument('--reasoning', action='store_true', help='Generate Chain-of-Thought <think> blocks in completions.')
  args = parser.parse_args()

  out_path = pathlib.Path(args.output_dir)
  out_path.mkdir(parents=True, exist_ok=True)

  game = hanabi_env.HanabiGame(players=2)
  dz = determinize.Determinizer(game, harvest_seed=args.seed)
  renderer = state_renderers.HanabiRenderer(max_history_turns=args.max_history_turns)
  expert = belief_expert.SafeBeliefLookaheadPlayer(game, dz, n_worlds=args.n_worlds, seed=args.seed)

  system_prompts = [
      build_system_prompt(game, p, reasoning=args.reasoning)
      for p in range(game.num_players())
  ]

  print(f'Starting BC generation (reasoning={args.reasoning}): {args.num_games} games, n_worlds={args.n_worlds}')
  records = []
  scores = []
  t0 = time.time()

  for g in range(args.num_games):
    g_seed = args.seed + g * 100
    random.seed(g_seed)
    state = game.new_initial_state()
    game_records = []
    turn = 0

    while not state.is_terminal() and turn < 150:
      player = state.current_player()
      if player < 0:
        break

      state_text = renderer.render_state(state, player, game)
      legal_actions_with_desc = renderer.render_legal_actions(state, player, game)
      action_descriptions = [d for _, d in legal_actions_with_desc]

      prompt = build_full_prompt(system_prompts[player], state_text, action_descriptions)

      action_id = expert.select_action(state, player, game)

      target_desc = None
      for a, d in legal_actions_with_desc:
        if a == action_id:
          target_desc = d
          break
      assert target_desc is not None, f'Action {action_id} not found in legal actions'

      if args.reasoning:
        completion = generate_cot_reasoning(state, player, target_desc, expert._bot)
      else:
        completion = target_desc

      # Round-trip verification:
      parsed_id = renderer.parse_action(completion, legal_actions_with_desc)
      assert parsed_id == action_id, f'Parse mismatch: {parsed_id} != {action_id} for "{completion}"'

      game_records.append({
          'game_id': g,
          'turn': turn,
          'player': player,
          'prompt': prompt,
          'completion': completion,
          'action_id': action_id,
      })

      state.apply_action(action_id)
      turn += 1

    final_score = state.score()
    scores.append(final_score)

    for r in game_records:
      r['final_score'] = final_score
      records.append(r)

    print(f'Game {g+1}/{args.num_games}: score={final_score}, turns={turn}, lives={state.life_tokens()}, total_samples={len(records)}')

  elapsed = time.time() - t0
  mean_score = sum(scores) / len(scores) if scores else 0
  print(f'\nGeneration finished in {elapsed:.1f}s ({elapsed/len(records):.3f}s/sample)')
  print(f'Total samples: {len(records)}, Mean game score: {mean_score:.2f}')

  # Shuffle game-level splits to avoid data leakage
  num_train_games = int(args.num_games * args.train_split)
  train_records = [r for r in records if r['game_id'] < num_train_games]
  val_records = [r for r in records if r['game_id'] >= num_train_games]

  train_file = out_path / 'train.jsonl'
  val_file = out_path / 'val.jsonl'

  with open(train_file, 'w') as f:
    for r in train_records:
      f.write(json.dumps(r) + '\n')

  with open(val_file, 'w') as f:
    for r in val_records:
      f.write(json.dumps(r) + '\n')

  meta_file = out_path / 'metadata.json'
  with open(meta_file, 'w') as f:
    json.dump({
        'num_games': args.num_games,
        'train_samples': len(train_records),
        'val_samples': len(val_records),
        'mean_score': mean_score,
        'scores': scores,
        'n_worlds': args.n_worlds,
        'elapsed_seconds': elapsed,
    }, f, indent=2)

  print(f'Saved {len(train_records)} train samples to {train_file}')
  print(f'Saved {len(val_records)} val samples to {val_file}')


if __name__ == '__main__':
  main()
