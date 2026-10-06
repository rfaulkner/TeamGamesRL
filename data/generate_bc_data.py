"""Generates Supervised Fine-Tuning (BC) data for Hanabi using Belief Expert."""

import argparse
import collections
import json
import multiprocessing
import pathlib
import random
import re
import sys
import time

import types
_stub = types.ModuleType('pyspiel')
_stub.State = object
_stub.Game = object
sys.modules.setdefault('pyspiel', _stub)

# Ensure the project root is importable when run as a blaze binary or script.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from env import state_renderers  # pylint: disable=g-import-not-at-top
from env.hanabi import belief_expert  # pylint: disable=g-import-not-at-top
from env.hanabi import determinize  # pylint: disable=g-import-not-at-top
from env.hanabi import hanabi_env  # pylint: disable=g-import-not-at-top

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
- Inside <think>...</think>, deliberate across these key aspects before choosing:
  1. Fireworks & Needed Cards: Note current firework heights and what ranks are needed next.
  2. Own Hand (Playability & Risk): Evaluate your card clues against needed fireworks and remaining lives. Is a card confirmed playable, or is playing it a gamble?
  3. Partner & Communication: Look at partner's visible cards and info tokens. Does partner need an immediate clue to play safely or avoid discarding a critical card?
  4. Token Management & Discards: If info tokens are low, identify safe discards (dead cards or duplicate ranks) to replenish tokens without risking vital cards.
  5. Decision: Weigh your options (Play, Hint, Discard) to balance scoring progress, communication, and team survival.
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
  """Generates a concise, evaluative reasoning block for the chosen action."""
  obs_string = state.observation_string(player_id)
  fireworks = bot._parse_fireworks(obs_string)
  info_tokens = state.information_tokens()
  lives = state.life_tokens()

  needed = {c: fireworks.get(c, 0) + 1 for c in ('B', 'G', 'R', 'W', 'Y') if fireworks.get(c, 0) < 5}
  needed_str = ', '.join(f'{c}{r}' for c, r in sorted(needed.items())) if needed else 'None'
  fw_str = ', '.join(f'{c}:{h}' for c, h in sorted(fireworks.items()))

  reasons = [
      f'State: Fireworks [{fw_str}], Needed [{needed_str}], Info {info_tokens}/8, Lives {lives}/3.',
  ]

  action_lower = target_desc.lower()
  if action_lower.startswith('play'):
    reasons.append(
        'Own hand: Card clues confirm this card matches a currently needed firework rank.'
    )
    reasons.append(
        'Evaluation: Playing is safe and directly advances the team score.'
    )
  elif action_lower.startswith('hint') or 'reveal' in action_lower or 'hint ' in action_lower:
    reasons.append(
        'Own hand: No cards confirmed safe to play without risking a life.'
    )
    reasons.append(
        f'Partner & Clues: Partner holds cards matching needed fireworks ({needed_str}); giving clue to enable safe play.'
    )
    reasons.append(
        'Evaluation: Providing information coordinates progress while keeping team safe.'
    )
  elif action_lower.startswith('discard'):
    reasons.append(
        'Own hand: No cards confirmed safe to play.'
    )
    reasons.append(
        f'Token management: Info tokens ({info_tokens}/8) need replenishment; identifying a safe, non-critical discard.'
    )
    reasons.append(
        'Evaluation: Discarding regains an info token for future communication without losing a unique card.'
    )
  else:
    reasons.append('Deliberating options across playability, communication, and token management.')

  reasons.append(f'Decision: {target_desc}.')
  think_body = '\n'.join(f'- {r}' for r in reasons)
  return f'<think>\n{think_body}\n</think>\n{target_desc}'


# ── Expert-grounded CoT ──────────────────────────────────────────────────────
# The templated block above is identical for every hint and every discard, so
# it cannot teach the model *which* hint or *which* card -- the two decisions
# both BC adapters get wrong most often.  The expert-grounded block instead
# writes down the facts the expert actually conditions on, all of which the
# model can recompute from the prompt, plus the expert's own value estimate
# (mean rollout score over its determinized worlds) for EVERY candidate:
#
#   Need: R2 Y1 G3 W2 B1 | info 4/8 | lives 3 | deck 22
#   Mine: c0 ?? c1 R? c2 ?1 c3 ?? c4 ?? -> none known playable
#   P1: c0 Y1* c1 R2 c2 G4 c3 W2* c4 B5! (*=playable now, !=last copy)
#   Hints to P1: Yellow (c0 Y1*, new) 12.3 | Red (c1 R2, new) 10.9 | ...
#       | rank 1 (c0 Y1*, new) 12.3 | rank 2 (c1 R2 c3 W2*, new) 11.7 | ...
#   Discards: c0 (unhinted, oldest) 11.0 | c1 (hinted Red) 9.8 | ...
#   Best action: Hint Player 1 about Yellow cards
#
# Candidates are listed in prompt order (ascending action id), never
# best-first: autoregressively, a best-first list would commit to the action
# at the first Options token and turn the values into post-hoc rationalisation.
# In prompt order the model must produce the values first and ``Best action``
# is the first candidate attaining the maximum -- exactly the expert's own
# tie-break (``select_action`` iterates hints then discards in legal order).
# Known-playable turns have no candidate lines (the expert never rolls out).

_COLOR_NAMES = {'R': 'Red', 'Y': 'Yellow', 'G': 'Green', 'W': 'White', 'B': 'Blue'}
_CARD_COPIES = {1: 3, 2: 2, 3: 2, 4: 2, 5: 1}
_REVEAL_COLOR_RE = re.compile(r'\(Reveal player \+(\d+) color ([RYGWB])\)')
_REVEAL_RANK_RE = re.compile(r'\(Reveal player \+(\d+) rank (\d+)\)')
_PLAY_RE = re.compile(r'\(Play (\d+)\)')
_DISCARD_RE = re.compile(r'\(Discard (\d+)\)')
# Own card: (hinted_color, hinted_rank, plausible_colors, plausible_ranks);
# 'X' marks an unhinted attribute.  Seen card: (color, rank, hinted_color,
# hinted_rank) for a partner's card we can look at.
_OwnCard = tuple[str, str, str, str]
_SeenCard = tuple[str, int, str, str]


def _split_knowledge(knowledge: str) -> _OwnCard:
  """'X3|RYGWB3' -> (hinted_color, hinted_rank, plausible_colors, plausible_ranks)."""
  hinted, _, plausible = knowledge.partition('|')
  hinted = (hinted + 'XX')[:2]
  p_colors = ''.join(ch for ch in plausible if ch in _COLOR_NAMES)
  p_ranks = ''.join(ch for ch in plausible if ch.isdigit())
  return hinted[0], hinted[1], p_colors, p_ranks


def _parse_hands(
    obs_string: str,
) -> tuple[list[_OwnCard], list[list[_SeenCard]]]:
  """Parses own-hand knowledge and the other players' visible hands.

  Args:
    obs_string: Raw observation string for the acting player.

  Returns:
    own: one ``_OwnCard`` per card, in hand order (index 0 = oldest).
    others: hands in relative order (next player first); each a list of
      ``_SeenCard``.
  """
  own: list[_OwnCard] = []
  others: list[list[_SeenCard]] = []
  current: list[_OwnCard | _SeenCard] = []
  is_own = False
  in_hands = False
  for raw in obs_string.split('\n'):
    line = raw.strip()
    if line == 'Hands:':
      in_hands = True
      continue
    if not in_hands:
      continue
    if line.startswith('Deck size:'):
      break
    if line == 'Cur player':
      is_own = True
      current = []
    elif line == '-----':
      if is_own:
        own = current
      else:
        others.append(current)
      current, is_own = [], False
    elif '||' in line:
      face, _, knowledge = line.partition('||')
      face, knowledge = face.strip(), knowledge.strip()
      hc, hr, pc, pr = _split_knowledge(knowledge)
      if face.startswith('X'):
        current.append((hc, hr, pc, pr))
      else:
        current.append((face[0], int(face[1]), hc, hr))
  return own, others


def _discard_counts(obs_string: str) -> dict[tuple[str, int], int]:
  counts: dict[tuple[str, int], int] = collections.Counter()
  for raw in obs_string.split('\n'):
    if raw.strip().startswith('Discards:'):
      for tok in raw.split(':', 1)[1].split():
        if len(tok) >= 2 and tok[0] in _COLOR_NAMES and tok[1].isdigit():
          counts[(tok[0], int(tok[1]))] += 1
  return counts


def _own_card_code(card: _OwnCard) -> str:
  """Own card as '<color><rank>' with '?' for anything not pinned down."""
  _, _, p_colors, p_ranks = card
  c = p_colors if len(p_colors) == 1 else '?'
  r = p_ranks if len(p_ranks) == 1 else '?'
  return f'{c}{r}'


def _own_known_playable(card: _OwnCard, fireworks: dict[str, int]) -> bool:
  _, _, p_colors, p_ranks = card
  return (
      len(p_colors) == 1
      and len(p_ranks) == 1
      and int(p_ranks) == fireworks.get(p_colors, 0) + 1
  )


def _own_known_dead(card: _OwnCard, fireworks: dict[str, int]) -> bool:
  _, _, p_colors, p_ranks = card
  return (
      len(p_colors) == 1
      and len(p_ranks) == 1
      and int(p_ranks) <= fireworks.get(p_colors, 0)
  )


def _visible_flags(
    color: str, rank: int, fireworks: dict[str, int], discards
) -> str:
  flags = ''
  if rank == fireworks.get(color, 0) + 1:
    flags += '*'
  if rank > fireworks.get(color, 0) and (
      discards.get((color, rank), 0) >= _CARD_COPIES.get(rank, 1) - 1
  ):
    flags += '!'
  return flags


def _hint_option(
    label: str,
    hand: list[_SeenCard],
    by_color: bool,
    key: str | int,
    fireworks: dict[str, int],
    discards,
) -> str:
  """'<label> (<touched cards>, new|known)' for a hint candidate."""
  touched, new_info = [], False
  for i, (color, rank, hc, hr) in enumerate(hand):
    if (color if by_color else rank) != key:
      continue
    touched.append(
        f'c{i} {color}{rank}{_visible_flags(color, rank, fireworks, discards)}'
    )
    already = (hc != 'X') if by_color else (hr != 'X')
    new_info = new_info or not already
  detail = ' '.join(touched) if touched else 'nothing'
  return f'{label} ({detail}, {"new" if new_info else "known"})'


def _discard_option(
    pos: int, own: list[_OwnCard], fireworks: dict[str, int]
) -> str:
  """'c<pos> (<what we know about it>)' for a discard candidate."""
  if pos >= len(own):
    return f'c{pos}'
  card = own[pos]
  hc, hr, _, _ = card
  if _own_known_dead(card, fireworks):
    return f'c{pos} (known {_own_card_code(card)}, dead)'
  if hc == 'X' and hr == 'X':
    unhinted = [i for i, c in enumerate(own) if c[0] == 'X' and c[1] == 'X']
    oldest = ', oldest' if unhinted and pos == min(unhinted) else ''
    return f'c{pos} (unhinted{oldest})'
  told = []
  if hc != 'X':
    told.append(_COLOR_NAMES[hc])
  if hr != 'X':
    told.append(f'rank {hr}')
  return f'c{pos} (hinted {" ".join(told)})'


def _candidate_lines(
    state,
    player_id: int,
    num_players: int,
    q_values: dict[int, float],
    own: list[_OwnCard],
    others: list[list[_SeenCard]],
    fireworks: dict[str, int],
    discards,
    info_tokens: int,
) -> tuple[list[str], list[int]]:
  """Formats every lookahead candidate with its value, in prompt order.

  Args:
    state: Current game state (for ``action_to_string``).
    player_id: Acting player.
    num_players: Number of players in the game.
    q_values: ``{action_id: mean rollout score}`` from the expert.
    own: Parsed own-hand knowledge.
    others: Parsed visible hands of the other players.
    fireworks: Current firework heights.
    discards: Discard-pile counts.
    info_tokens: Current information tokens (for the "none" placeholders).

  Returns:
    (lines, listing_order): the ``Hints``/``Discards`` lines, and the action
    ids in the exact order they are listed, so the caller can verify that the
    expert's choice is the first maximum of that listing.
  """
  hints: list[tuple[int, str]] = []
  discard_opts: list[tuple[int, str]] = []
  other_opts: list[tuple[int, str]] = []
  for a in sorted(q_values):  # ascending action id == legal-action order
    s = state.action_to_string(player_id, a)
    q = f'{q_values[a]:.1f}'
    m = _REVEAL_COLOR_RE.match(s) or _REVEAL_RANK_RE.match(s)
    if m:
      offset = int(m.group(1))
      hand = others[offset - 1] if 0 < offset <= len(others) else []
      by_color = 'color' in s
      key: str | int = m.group(2) if by_color else int(m.group(2))
      label = _COLOR_NAMES[m.group(2)] if by_color else f'rank {m.group(2)}'
      if num_players > 2:
        label = f'P{(player_id + offset) % num_players} {label}'
      hints.append(
          (a, f'{_hint_option(label, hand, by_color, key, fireworks, discards)} {q}')
      )
      continue
    m = _DISCARD_RE.match(s)
    if m:
      discard_opts.append((a, f'{_discard_option(int(m.group(1)), own, fireworks)} {q}'))
      continue
    m = _PLAY_RE.match(s)
    other_opts.append((a, f'{"play c" + m.group(1) if m else s} (gamble) {q}'))

  hint_hdr = f'Hints to P{(player_id + 1) % num_players}' if num_players == 2 else 'Hints'
  none = f'none (info {info_tokens}/8)'
  lines = [
      f'{hint_hdr}: ' + (' | '.join(t for _, t in hints) if hints else none),
      'Discards: ' + (' | '.join(t for _, t in discard_opts) if discard_opts else none),
  ]
  if other_opts:
    lines.append('Plays: ' + ' | '.join(t for _, t in other_opts))
  order = [a for a, _ in hints] + [a for a, _ in discard_opts] + [a for a, _ in other_opts]
  return lines, order


def generate_expert_cot(
    state,
    player_id: int,
    target_desc: str,
    expert,
    num_players: int,
) -> str:
  """Builds the expert-grounded <think> block for the expert's last decision."""
  decision = dict(expert.last_decision)
  obs_string = state.observation_string(player_id)
  fireworks = expert._bot._parse_fireworks(obs_string)
  info_tokens = state.information_tokens()
  lives = state.life_tokens()
  deck = state.deck_size()
  discards = _discard_counts(obs_string)
  own, others = _parse_hands(obs_string)

  need = ' '.join(
      f'{c}{fireworks.get(c, 0) + 1}' for c in 'RYGWB' if fireworks.get(c, 0) < 5
  ) or 'none'
  lines = [f'Need: {need} | info {info_tokens}/8 | lives {lives} | deck {deck}']

  kind = decision.get('kind')
  mine = ' '.join(f'c{i} {_own_card_code(c)}' for i, c in enumerate(own))
  if kind == 'known_playable':
    pos = decision.get('playable_position')
    card = own[pos] if pos is not None and pos < len(own) else None
    if card is not None:
      lines.append(
          f'Mine: {mine} -> c{pos} is {_COLOR_NAMES[card[2]]} {card[3]}'
          ' = needed: safe play'
      )
    else:
      lines.append(f'Mine: {mine} -> known playable card: safe play')
  else:
    lines.append(f'Mine: {mine} -> none known playable')
    for offset, hand in enumerate(others, start=1):
      pid = (player_id + offset) % num_players
      cards = ' '.join(
          f'c{i} {color}{rank}{_visible_flags(color, rank, fireworks, discards)}'
          for i, (color, rank, _, _) in enumerate(hand)
      )
      lines.append(f'P{pid}: {cards} (*=playable now, !=last copy)')

  if kind == 'lookahead':
    q_values = decision.get('q_values', {})
    chosen = decision['action']
    cand_lines, order = _candidate_lines(
        state, player_id, num_players, q_values, own, others, fireworks,
        discards, info_tokens,
    )
    lines.extend(cand_lines)
    # ``max`` returns the first maximal element, i.e. the first listed
    # candidate attaining the best value -- this must be the expert's choice,
    # otherwise the written values would not justify the written action.
    first_max = max(order, key=lambda a: q_values[a])
    assert first_max == chosen, (
        f'listing order tie-break mismatch: first max {first_max}, '
        f'expert chose {chosen}, q={q_values}'
    )
  elif kind == 'single_legal':
    lines.append('Only legal action.')
  elif kind == 'single_candidate':
    lines.append('Only safe option (no gamble on an unknown card).')
  elif kind == 'heuristic_fallback':
    lines.append('No consistent world found; falling back to safe-play heuristic.')

  lines.append(f'Best action: {target_desc}')
  return '<think>\n' + '\n'.join(lines) + f'\n</think>\n{target_desc}'


def build_full_prompt(system_prompt: str, state_text: str, action_descriptions: list[str]) -> str:
  actions_lines = [f'  - {desc}' for desc in action_descriptions]
  actions_text = '\n'.join(actions_lines)
  user_prompt = _USER_PROMPT_TEMPLATE.format(
      state_text=state_text,
      actions_text=actions_text,
  )
  return f'{system_prompt}\n{user_prompt}'


def _hle_seed(base_seed: int, shard_idx: int) -> int:
  """HLE deal seed for a worker shard.

  Each worker owns one ``HanabiGame`` whose RNG persists across games, so the
  shard seed fixes a reproducible, disjoint sequence of deals per worker.  The
  range is kept far from the RL trainer's fixed evaluation deals (``eval_seed``
  10000..10099) so training data can never contain an eval deal.
  """
  return 20000 + base_seed + 1000 * shard_idx


def action_type(desc: str) -> str:
  """Coarse action class of a rendered action description."""
  lower = desc.lower()
  if lower.startswith('play'):
    return 'play'
  if lower.startswith('discard'):
    return 'discard'
  if lower.startswith('hint') or 'reveal' in lower:
    return 'hint'
  return 'other'


def _run_shard(spec: dict) -> dict:
  """Plays a contiguous block of games with a private expert and game RNG.

  Runs in a worker process.  Returns the records plus per-game scores/turn
  counts and action-type counts so the parent can merge shards.
  """
  shard_idx = spec['shard_idx']
  game_ids = spec['game_ids']
  seed = spec['seed']
  reasoning = spec['reasoning']
  reasoning_mode = spec.get('reasoning_mode', 'templated')
  shard_seed = seed + 1000003 * shard_idx

  game = hanabi_env.HanabiGame(players=2, seed=_hle_seed(seed, shard_idx))
  dz = determinize.Determinizer(game, harvest_seed=shard_seed)
  renderer = state_renderers.HanabiRenderer(
      max_history_turns=spec['max_history_turns']
  )
  expert = belief_expert.SafeBeliefLookaheadPlayer(
      game, dz, n_worlds=spec['n_worlds'], seed=shard_seed
  )
  system_prompts = [
      build_system_prompt(game, p, reasoning=reasoning)
      for p in range(game.num_players())
  ]

  records = []
  scores = {}
  turns_by_game = {}
  type_counts = collections.Counter()
  t0 = time.time()

  for n, g in enumerate(game_ids):
    random.seed(seed + g * 100)
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

      if reasoning and reasoning_mode == 'expert':
        completion = generate_expert_cot(
            state, player, target_desc, expert, game.num_players()
        )
      elif reasoning:
        completion = generate_cot_reasoning(state, player, target_desc, expert._bot)
      else:
        completion = target_desc

      # Round-trip verification:
      parsed_id = renderer.parse_action(completion, legal_actions_with_desc)
      assert parsed_id == action_id, f'Parse mismatch: {parsed_id} != {action_id} for "{completion}"'

      type_counts[action_type(target_desc)] += 1
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
    scores[g] = final_score
    turns_by_game[g] = turn
    for r in game_records:
      r['final_score'] = final_score
    records.extend(game_records)

    print(
        f'[shard {shard_idx}] game {g} ({n + 1}/{len(game_ids)}): '
        f'score={final_score}, turns={turn}, lives={state.life_tokens()}, '
        f'shard_samples={len(records)}, elapsed={time.time() - t0:.0f}s',
        flush=True,
    )

  return {
      'shard_idx': shard_idx,
      'hle_seed': _hle_seed(seed, shard_idx),
      'records': records,
      'scores': scores,
      'turns': turns_by_game,
      'type_counts': dict(type_counts),
  }


def main():
  parser = argparse.ArgumentParser(description='Generate BC data for Hanabi.')
  parser.add_argument('--num_games', type=int, default=50, help='Number of self-play games to run.')
  parser.add_argument('--n_worlds', type=int, default=3, help='Determinized worlds per decision.')
  parser.add_argument('--max_history_turns', type=int, default=20, help='History turns in renderer.')
  parser.add_argument('--output_dir', type=str, default='data/bc_hanabi', help='Directory to write JSONL.')
  parser.add_argument('--train_split', type=float, default=0.9, help='Fraction of data for training.')
  parser.add_argument('--seed', type=int, default=42, help='Base random seed.')
  parser.add_argument('--reasoning', action='store_true', help='Generate Chain-of-Thought <think> blocks in completions.')
  parser.add_argument(
      '--reasoning_mode',
      choices=('templated', 'expert'),
      default='templated',
      help=(
          'With --reasoning: "templated" emits the fixed per-action-type '
          'rationale; "expert" emits a grounded state summary plus the '
          "expert's own candidate ranking (see generate_expert_cot)."
      ),
  )
  parser.add_argument(
      '--num_workers',
      type=int,
      default=1,
      help=(
          'Worker processes.  Games are split into contiguous blocks; each '
          'worker owns its own expert, determinizer and (seeded) deal RNG.'
      ),
  )
  args = parser.parse_args()

  out_path = pathlib.Path(args.output_dir)
  out_path.mkdir(parents=True, exist_ok=True)

  num_workers = max(1, min(args.num_workers, args.num_games))
  bounds = [round(i * args.num_games / num_workers) for i in range(num_workers + 1)]
  specs = [
      {
          'shard_idx': i,
          'game_ids': list(range(bounds[i], bounds[i + 1])),
          'seed': args.seed,
          'n_worlds': args.n_worlds,
          'max_history_turns': args.max_history_turns,
          'reasoning': args.reasoning,
          'reasoning_mode': args.reasoning_mode,
      }
      for i in range(num_workers)
      if bounds[i + 1] > bounds[i]
  ]

  reasoning_mode = args.reasoning_mode if args.reasoning else None
  print(
      f'Starting BC generation (reasoning={args.reasoning}, '
      f'mode={reasoning_mode}): {args.num_games} games, '
      f'n_worlds={args.n_worlds}, workers={len(specs)}',
      flush=True,
  )
  t0 = time.time()
  if len(specs) == 1:
    results = [_run_shard(specs[0])]
  else:
    with multiprocessing.get_context('fork').Pool(processes=len(specs)) as pool:
      results = pool.map(_run_shard, specs)

  records = []
  scores = {}
  turns = {}
  type_counts = collections.Counter()
  for r in results:
    records.extend(r['records'])
    scores.update(r['scores'])
    turns.update(r['turns'])
    type_counts.update(r['type_counts'])
  records.sort(key=lambda r: (r['game_id'], r['turn']))
  score_list = [scores[g] for g in range(args.num_games)]
  turn_list = [turns[g] for g in range(args.num_games)]

  elapsed = time.time() - t0
  mean_score = sum(score_list) / len(score_list) if score_list else 0
  total = sum(type_counts.values()) or 1
  type_fracs = {k: type_counts[k] / total for k in ('play', 'discard', 'hint', 'other')}
  print(f'\nGeneration finished in {elapsed:.1f}s ({elapsed / max(len(records), 1):.3f}s/sample)')
  print(f'Total samples: {len(records)}, Mean game score: {mean_score:.2f}, Mean turns: {sum(turn_list) / len(turn_list):.1f}')
  print('Action mix: ' + ', '.join(f'{k}={type_counts[k]} ({type_fracs[k]:.1%})' for k in type_fracs))

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
        'mean_turns': sum(turn_list) / len(turn_list) if turn_list else 0,
        'scores': score_list,
        'n_worlds': args.n_worlds,
        'max_history_turns': args.max_history_turns,
        'reasoning': args.reasoning,
        'reasoning_mode': reasoning_mode,
        'seed': args.seed,
        'num_workers': len(specs),
        'hle_seeds': [r['hle_seed'] for r in results],
        'action_type_counts': dict(type_counts),
        'action_type_fracs': type_fracs,
        'elapsed_seconds': elapsed,
    }, f, indent=2)

  print(f'Saved {len(train_records)} train samples to {train_file}')
  print(f'Saved {len(val_records)} val samples to {val_file}')


if __name__ == '__main__':
  main()
