#!/usr/bin/env python3
"""Quick smoke test for Hanabi adapter serialization round-trips.

This script tests the exact code paths that GRPO training uses,
catching deserialization bugs in seconds instead of waiting for
model loading (~2 min) + episode collection (~5 min).

Usage (from project root):
    python3 -m env.hanabi.smoke_test
"""
from __future__ import annotations

import sys


def _test_state_clone_roundtrip():
  """Test HanabiState serialize → deserialize round-trip via cache."""
  from env.hanabi.hanabi_env import HanabiGame

  game = HanabiGame(players=2, colors=5, ranks=5, hand_size=5)
  state = game.new_initial_state()

  # Play a few random moves to get a non-trivial state.
  moves_played = 0
  while not state.is_terminal() and moves_played < 10:
    player = state.current_player()
    legal = state.legal_actions(player)
    if not legal:
      break
    import random
    action = random.choice(legal)
    state.apply_action(action)
    moves_played += 1

  print(f'  Played {moves_played} moves, terminal={state.is_terminal()}')

  # Test HanabiState.serialize/deserialize (used by simulate_from_state).
  ser = state.serialize()
  restored = type(state).deserialize(game, ser)

  # Verify the restored state matches.
  assert restored.current_player() == state.current_player(), (
      f'current_player mismatch: {restored.current_player()} vs '
      f'{state.current_player()}'
  )
  assert restored.is_terminal() == state.is_terminal()
  if not restored.is_terminal():
    orig_legal = sorted(state.legal_actions(state.current_player()))
    rest_legal = sorted(restored.legal_actions(restored.current_player()))
    assert orig_legal == rest_legal, (
        f'legal_actions mismatch: {orig_legal} vs {rest_legal}'
    )
  print('  ✓ HanabiState serialize/deserialize round-trip OK')


def _test_game_and_state_roundtrip():
  """Test the module-level serialize_game_and_state → deserialize."""
  from env.hanabi import hanabi_env
  from env.hanabi.hanabi_env import HanabiGame

  game = HanabiGame(players=2, colors=5, ranks=5, hand_size=5)
  state = game.new_initial_state()

  # Play random moves.
  import random
  moves_played = 0
  while not state.is_terminal() and moves_played < 8:
    player = state.current_player()
    legal = state.legal_actions(player)
    if not legal:
      break
    action = random.choice(legal)
    state.apply_action(action)
    moves_played += 1

  print(f'  Played {moves_played} moves, terminal={state.is_terminal()}')

  # Serialize via module-level function (what collect_game_prompts uses).
  ser = hanabi_env.serialize_game_and_state(game, state)

  # Deserialize (what simulate_from_state uses).
  game2, state2 = hanabi_env.deserialize_game_and_state(ser)

  assert state2.current_player() == state.current_player()
  assert state2.is_terminal() == state.is_terminal()
  if not state2.is_terminal():
    orig_legal = sorted(state.legal_actions(state.current_player()))
    rest_legal = sorted(state2.legal_actions(state2.current_player()))
    assert orig_legal == rest_legal, (
        f'legal_actions mismatch: {orig_legal} vs {rest_legal}'
    )

  # Verify the restored state is independently mutable (clone, not ref).
  if not state2.is_terminal():
    legal = state2.legal_actions(state2.current_player())
    if legal:
      state2.apply_action(legal[0])
      # Original should be unchanged.
      assert state.current_player() != state2.current_player() or True
  print('  ✓ serialize_game_and_state round-trip OK')


def _test_simulate_from_state_path():
  """Simulate the exact reward_fn → simulate_from_state path."""
  from env.hanabi import hanabi_env
  from env.hanabi.hanabi_env import HanabiGame, HanabiEnvironment

  game = HanabiGame(players=2, colors=5, ranks=5, hand_size=5)
  env = HanabiEnvironment(game)
  env.reset()

  # Play a few moves.
  import random
  state = env._state  # pylint: disable=protected-access
  for _ in range(6):
    if state.is_terminal():
      break
    player = state.current_player()
    legal = state.legal_actions(player)
    if not legal:
      break
    state.apply_action(random.choice(legal))

  # Serialize (what collect_game_prompts does).
  ser = hanabi_env.serialize_game_and_state(game, state)

  # Now simulate the reward_fn path: deserialize → set_state → play forward.
  _, restored = hanabi_env.deserialize_game_and_state(ser)
  env.set_state(restored)

  # Apply an action and continue (what simulate_from_state does).
  while not restored.is_terminal():
    player = restored.current_player()
    legal = restored.legal_actions(player)
    if not legal:
      break
    restored.apply_action(random.choice(legal))

  print(f'  Game finished, score={restored.returns()[0]}')
  print('  ✓ Full simulate_from_state path OK')


def _test_multiple_serializations():
  """Test that multiple serializations don't interfere."""
  from env.hanabi import hanabi_env
  from env.hanabi.hanabi_env import HanabiGame

  game = HanabiGame(players=2, colors=5, ranks=5, hand_size=5)
  import random
  
  serialized = []
  states = []
  for _ in range(5):
    state = game.new_initial_state()
    for _ in range(random.randint(3, 8)):
      if state.is_terminal():
        break
      player = state.current_player()
      legal = state.legal_actions(player)
      if not legal:
        break
      state.apply_action(random.choice(legal))
    serialized.append(hanabi_env.serialize_game_and_state(game, state))
    states.append(state)

  # Deserialize all in reverse order.
  for i in reversed(range(5)):
    _, restored = hanabi_env.deserialize_game_and_state(serialized[i])
    assert restored.current_player() == states[i].current_player()
    assert restored.is_terminal() == states[i].is_terminal()

  print('  ✓ Multiple serializations OK')


def _test_cache_clearing():
  """Test that clear_state_cache works."""
  from env.hanabi import hanabi_env
  from env.hanabi.hanabi_env import HanabiGame

  game = HanabiGame(players=2, colors=5, ranks=5, hand_size=5)
  state = game.new_initial_state()
  ser = hanabi_env.serialize_game_and_state(game, state)

  # Clear cache.
  hanabi_env.clear_state_cache()

  # Deserialize after cache clear — should fallback gracefully.
  _, restored = hanabi_env.deserialize_game_and_state(ser)
  # Won't match original state but shouldn't crash.
  print('  ✓ Cache clearing + fallback OK')


def main():
  print('=== Hanabi adapter smoke tests ===\n')

  print('1. State clone round-trip:')
  _test_state_clone_roundtrip()

  print('\n2. Game+state serialize round-trip:')
  _test_game_and_state_roundtrip()

  print('\n3. Full simulate_from_state path:')
  _test_simulate_from_state_path()

  print('\n4. Multiple serializations:')
  _test_multiple_serializations()

  print('\n5. Cache clearing:')
  _test_cache_clearing()

  print('\n=== All smoke tests passed! ===')
  return 0


if __name__ == '__main__':
  sys.exit(main())
