#!/usr/bin/env python3
"""Tests for env/hanabi/determinize.py against real HLE.

The decisive test is the last one: a determinized state must produce a
byte-identical observation string for the acting player.  If it does, the
expert searching over determinized worlds is provably using only information
the policy also has.
"""

import collections
import importlib.util
import pathlib
import random
import statistics
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]


def load(name, relpath):
  spec = importlib.util.spec_from_file_location(name, REPO / relpath)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod
  spec.loader.exec_module(mod)
  return mod


hanabi_env = load('hanabi_env', 'env/hanabi/hanabi_env.py')
det_mod = load('determinize', 'env/hanabi/determinize.py')
from hanabi_learning_environment import pyhanabi  # pylint: disable=g-import-not-at-top

MT = pyhanabi.HanabiMoveType
FAILS = []


def check(cond, label, detail=''):
  print(f'{"PASS" if cond else "FAIL"}  {label}' + (f'  {detail}' if detail
                                                    else ''))
  if not cond:
    FAILS.append(label)


def build_position(game, seed, n_moves=26):
  """A mid-game position with real hint constraints, at a player node."""
  rng = random.Random(seed)
  state = game.new_initial_state()
  hle = state._hle_state
  moves = 0
  while moves < n_moves and not hle.is_terminal():
    if hle.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
      hle.deal_random_card()
      continue
    legal = hle.legal_moves()
    hints = [m for m in legal if m.type() in (MT.REVEAL_COLOR, MT.REVEAL_RANK)]
    discards = [m for m in legal if m.type() == MT.DISCARD]
    pick = (hints if hints and rng.random() < 0.65 else discards) or legal
    hle.apply_move(rng.choice(pick))
    moves += 1
  while not hle.is_terminal() and hle.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
    hle.deal_random_card()
  return state if (not hle.is_terminal() and hle.cur_player() >= 0) else None


game = hanabi_env.HanabiGame(players=2)
print('building determinizer (harvesting deal moves)...')
t0 = time.time()
dz = det_mod.Determinizer(game)
print(f'  harvested in {time.time() - t0:.2f}s\n')

check(len(dz._deal_moves) == 25, 'harvested all 25 deal moves',
      f'got {len(dz._deal_moves)}')

# ---------------------------------------------------------------------------
print('\n--- core properties over several positions ---')
# ---------------------------------------------------------------------------
rng = random.Random(4)
accept_rates, times, distinct_fracs = [], [], []
positions_tested = 0

for seed in range(8):
  state = build_position(game, seed)
  if state is None:
    continue
  positions_tested += 1
  player = state.current_player()
  hle = state._hle_state
  true_hand = tuple((c.color(), c.rank()) for c in hle.player_hands()[player])
  orig_obs = state.observation_string(player)
  orig_public = dz._public_signature(hle, player)
  orig_knowledge = dz._knowledge_signature(hle, player)

  worlds, obs_mismatch, know_mismatch, pub_mismatch = [], 0, 0, 0
  t0 = time.time()
  n = 40
  for _ in range(n):
    w = dz.determinize(state, player, rng)
    if w is None:
      continue
    whle = w._hle_state
    worlds.append(tuple((c.color(), c.rank())
                        for c in whle.player_hands()[player]))
    if w.observation_string(player) != orig_obs:
      obs_mismatch += 1
    if dz._knowledge_signature(whle, player) != orig_knowledge:
      know_mismatch += 1
    if dz._public_signature(whle, player) != orig_public:
      pub_mismatch += 1
  dt = time.time() - t0

  accept_rates.append(len(worlds) / n)
  times.append(1000 * dt / n)
  distinct_fracs.append(len(set(worlds)) / max(len(worlds), 1))

  ok = (len(worlds) == n and obs_mismatch == 0 and know_mismatch == 0
        and pub_mismatch == 0)
  print(f'  seed {seed}: player={player} accepted={len(worlds)}/{n} '
        f'distinct={len(set(worlds)):2d} '
        f'obs_mismatch={obs_mismatch} know_mismatch={know_mismatch} '
        f'pub_mismatch={pub_mismatch} {1000 * dt / n:.1f}ms/world '
        f'{"OK" if ok else "<-- PROBLEM"}')
  if not ok:
    FAILS.append(f'position seed {seed}')

check(positions_tested >= 5, 'built enough test positions',
      f'{positions_tested}')
check(min(accept_rates) == 1.0, 'every determinization accepted',
      f'min rate {min(accept_rates):.2f}')
check(statistics.mean(times) < 25, 'determinization is fast enough',
      f'{statistics.mean(times):.1f} ms mean')
check(statistics.mean(distinct_fracs) > 0.5,
      'sampled worlds are diverse (a real belief, not one world)',
      f'{statistics.mean(distinct_fracs):.2f} distinct fraction')

# ---------------------------------------------------------------------------
print('\n--- the honesty property: prompt must be unchanged ---')
# ---------------------------------------------------------------------------
state = build_position(game, 11)
player = state.current_player()
orig_obs = state.observation_string(player)
w = dz.determinize(state, player, rng)
check(w is not None, 'determinization succeeded for the honesty check')
if w is not None:
  same_obs = w.observation_string(player) == orig_obs
  check(same_obs, "acting player's observation string is byte-identical")
  # and it should differ for the OTHER player, who can see the changed hand
  other = 1 - player
  differs = w.observation_string(other) != state.observation_string(other)
  hand_changed = (
      tuple((c.color(), c.rank())
            for c in w._hle_state.player_hands()[player])
      != tuple((c.color(), c.rank())
               for c in state._hle_state.player_hands()[player]))
  check(not hand_changed or differs,
        "partner's view does change when the hand really changed",
        f'hand_changed={hand_changed} partner_view_differs={differs}')

# ---------------------------------------------------------------------------
print('\n--- the true world must lie in the sampled support ---')
# ---------------------------------------------------------------------------
# Checking for the exact 5-card joint hand is far too strict: the consistent
# set is typically thousands of worlds, so the true one is rarely drawn in a
# few hundred samples.  The meaningful property is that the sampler is not
# *excluding* the truth -- i.e. each slot's real card appears in that slot's
# marginal support.
state = build_position(game, 21)
player = state.current_player()
true_hand = tuple((c.color(), c.rank())
                  for c in state._hle_state.player_hands()[player])
draws = 600
per_slot = [collections.Counter() for _ in true_hand]
joint = collections.Counter()
for _ in range(draws):
  w = dz.determinize(state, player, rng)
  if w is None:
    continue
  hand = tuple((c.color(), c.rank())
               for c in w._hle_state.player_hands()[player])
  joint[hand] += 1
  for i, cr in enumerate(hand):
    per_slot[i][cr] += 1

missing = [i for i, cr in enumerate(true_hand) if per_slot[i][cr] == 0]
for i, cr in enumerate(true_hand):
  print(f'    slot {i}: true={cr} sampled {per_slot[i][cr]:3d}/{draws} '
        f'({len(per_slot[i])} distinct values in support)')
check(not missing,
      "every slot's true card is in the sampler's support",
      f'missing slots: {missing}' if missing else '')
print(f'    joint: {len(joint)} distinct hands in {draws} draws '
      f'-> support is large, so an exact-joint hit is not expected')

# ---------------------------------------------------------------------------
print('\n--- determinizing must not mutate the original ---')
# ---------------------------------------------------------------------------
state = build_position(game, 31)
player = state.current_player()
before = (state.score(), state.life_tokens(), state.information_tokens(),
          state.deck_size(),
          tuple(tuple((c.color(), c.rank()) for c in h)
                for h in state._hle_state.player_hands()))
for _ in range(10):
  dz.determinize(state, player, rng)
after = (state.score(), state.life_tokens(), state.information_tokens(),
         state.deck_size(),
         tuple(tuple((c.color(), c.rank()) for c in h)
               for h in state._hle_state.player_hands()))
check(before == after, 'original state untouched by determinization')

print()
if FAILS:
  print(f'{len(FAILS)} FAILURES: {FAILS}')
  sys.exit(1)
print('all checks passed')
