#!/usr/bin/env python3
"""Measure the real SafePlayPlayer baseline against real HLE, both conventions.

Everything quoted so far for the heuristic baseline came from a reimplementation
in scratch/probe_sim.py.  Now that HLE builds locally, this runs the ACTUAL
env/hanabi/heuristic_player.py against the ACTUAL env/hanabi/hanabi_env.py, so
the number is the one the training pipeline really sees.

It also validates the discard-order fix by running both settings of
``new_cards_at_index_zero`` over the same deals.

Run with HLE on PYTHONPATH under the venv.
"""

import argparse
import importlib.util
import pathlib
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]


def load(name, relpath):
  """Load a module by file path.

  env/__init__.py pulls in pyspiel transitively, so the submodules have to be
  loaded directly rather than imported as a package.
  """
  spec = importlib.util.spec_from_file_location(name, REPO / relpath)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod
  spec.loader.exec_module(mod)
  return mod


hanabi_env = load('hanabi_env', 'env/hanabi/hanabi_env.py')
heuristic = load('heuristic_player', 'env/hanabi/heuristic_player.py')


def play_one(game, seed, new_cards_at_index_zero):
  """Play one full self-play game; return (score, turns, lives, bombed)."""
  p0 = heuristic.SafePlayPlayer(
      seed=seed, new_cards_at_index_zero=new_cards_at_index_zero)
  p1 = heuristic.SafePlayPlayer(
      seed=seed + 100000, new_cards_at_index_zero=new_cards_at_index_zero)
  players = [p0, p1]

  state = game.new_initial_state()
  turns = 0
  while not state.is_terminal() and turns < 300:
    cur = state.current_player()
    if cur < 0:
      break
    action = players[cur].select_action(state, cur, game)
    state.apply_action(action)
    turns += 1
  return state.score(), turns, state.life_tokens(), state.life_tokens() == 0


def run(game, n, new_cards_at_index_zero):
  scores, turns, bombs = [], [], 0
  for i in range(n):
    sc, tn, _, bombed = play_one(game, 1000 + i, new_cards_at_index_zero)
    scores.append(sc)
    turns.append(tn)
    bombs += bombed
  return scores, turns, bombs


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--games', type=int, default=200)
  args = ap.parse_args()

  game = hanabi_env.HanabiGame(players=2)
  print(f'Real HLE, real SafePlayPlayer, {args.games} self-play games each.\n')

  results = {}
  for label, flag in [
      ('FIXED   index 0 = oldest (matches HLE)', False),
      ('LEGACY  index 0 = newest (pre-fix behaviour)', True),
  ]:
    sc, tn, bm = run(game, args.games, flag)
    results[flag] = sc
    print(f'{label}')
    print(f'   mean score  {statistics.mean(sc):6.2f} / 25  '
          f'(sd {statistics.pstdev(sc):.2f})')
    print(f'   median      {statistics.median(sc):6.1f}')
    print(f'   min / max   {min(sc)} / {max(sc)}')
    print(f'   bombed out  {bm:4d}/{args.games} ({100 * bm / args.games:.1f}%)')
    print(f'   mean turns  {statistics.mean(tn):6.1f}')
    print()

  a, b = results[False], results[True]
  diff = [x - y for x, y in zip(a, b)]
  md = statistics.mean(diff)
  se = statistics.pstdev(diff) / (len(diff) ** 0.5)
  print(f'paired difference (fixed - legacy) = {md:+.3f} +/- {1.96 * se:.3f} '
        '(95% CI)')
  print(f'fixed better in {sum(1 for d in diff if d > 0)}/{len(diff)} deals, '
        f'worse in {sum(1 for d in diff if d < 0)}')


if __name__ == '__main__':
  main()
