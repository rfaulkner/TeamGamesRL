#!/usr/bin/env python3
"""Viewer for TeamGamesRL episode logs.

Reads the JSONL episode log produced by gemma_rl_trainer.py and displays
game states, LLM responses, and actions in a human-readable format.

Usage:
  # View all logged episodes
  python view_episodes.py --log_path=/scratch/user/teamgamesrl/run/episode_log.jsonl

  # View specific episodes
  python view_episodes.py --log_path=episode_log.jsonl --episodes=10,50,100

  # View only the last N episodes
  python view_episodes.py --log_path=episode_log.jsonl --last=5

  # Show full prompts (verbose)
  python view_episodes.py --log_path=episode_log.jsonl --show_prompts --last=3

  # Summary table only (no step details)
  python view_episodes.py --log_path=episode_log.jsonl --summary_only
"""

import argparse
import json
import sys

def parse_args():
  parser = argparse.ArgumentParser(
      description='View TeamGamesRL episode logs.')
  parser.add_argument(
      '--log_path', required=True,
      help='Path to the episode_log.jsonl file.')
  parser.add_argument(
      '--episodes', default='',
      help='Comma-separated list of episode numbers to display.')
  parser.add_argument(
      '--last', type=int, default=0,
      help='Show only the last N episodes.')
  parser.add_argument(
      '--show_prompts', action='store_true',
      help='Show the full LLM prompts (can be very verbose).')
  parser.add_argument(
      '--summary_only', action='store_true',
      help='Show only the summary table, no step-by-step details.')
  return parser.parse_args()

# ── ANSI colors ──────────────────────────────────────────────────────────────

RESET = '\033[0m'
BOLD = '\033[1m'
DIM = '\033[2m'

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'

PLAYER_COLORS = [BLUE, MAGENTA, GREEN, YELLOW]


def _color(text, color):
  return f'{color}{text}{RESET}'


def _player_color(player_id):
  return PLAYER_COLORS[player_id % len(PLAYER_COLORS)]


# ── Loading ──────────────────────────────────────────────────────────────────


def load_episodes(log_path):
  """Loads all episodes from a JSONL file."""
  episodes = []
  with open(log_path, 'r') as f:
    for line_num, line in enumerate(f, 1):
      line = line.strip()
      if not line:
        continue
      try:
        episodes.append(json.loads(line))
      except json.JSONDecodeError as e:
        print(f'Warning: skipping malformed line {line_num}: {e}',
              file=sys.stderr)
  return episodes


# ── Display ──────────────────────────────────────────────────────────────────


def print_summary_table(episodes):
  """Prints a compact summary table of all episodes."""
  print()
  print(_color('Episode Summary', BOLD))
  print('=' * 80)

  # Header
  header = (
      f'{"Ep":>5}  {"Type":<6}  {"Loss":>10}  '
  )
  # Add per-player reward columns based on first episode
  if episodes:
    num_players = len(episodes[0].get('players', []))
    for p in range(num_players):
      header += f'{"P" + str(p) + " Reward":>10}  '
    header += f'{"Steps":>6}'
  print(_color(header, DIM))
  print('-' * 80)

  for ep in episodes:
    ep_type = 'EVAL' if ep.get('is_evaluation') else 'TRAIN'
    loss = ep.get('loss', 0.0)
    row = f'{ep["episode"]:>5}  {ep_type:<6}  {loss:>10.4f}  '

    players = ep.get('players', [])
    total_steps = 0
    for p_data in players:
      reward = p_data.get('reward', 0.0)
      color = GREEN if reward > 0 else (RED if reward < 0 else DIM)
      row += _color(f'{reward:>10.2f}', color) + '  '
      total_steps += len(p_data.get('steps', []))

    row += f'{total_steps:>6}'
    print(row)

  print('=' * 80)
  print(f'Total episodes logged: {len(episodes)}')
  print()


def print_episode_detail(ep, show_prompts=False):
  """Prints detailed step-by-step view of a single episode."""
  ep_num = ep['episode']
  game = ep.get('game', '?')
  ep_type = 'EVAL' if ep.get('is_evaluation') else 'TRAIN'
  loss = ep.get('loss', 0.0)

  print()
  print(_color(f'╔{"═" * 78}╗', BOLD))
  title = f' Episode {ep_num} ({ep_type}) — {game}'
  print(_color(f'║{title:<78}║', BOLD))
  print(_color(f'╚{"═" * 78}╝', BOLD))

  players = ep.get('players', [])
  rewards_str = ', '.join(
      _color(f'P{p["player_id"]}={p["reward"]:.1f}',
             _player_color(p['player_id']))
      for p in players
  )
  print(f'  Rewards: {rewards_str}   Loss: {loss:.4f}')
  print()

  # Interleave steps across players in order for a timeline view.
  # Collect all steps with their player info.
  all_steps = []
  for p_data in players:
    pid = p_data['player_id']
    for step_idx, step in enumerate(p_data.get('steps', [])):
      all_steps.append((pid, step_idx, step))

  if not all_steps:
    print(_color('  (No steps recorded)', DIM))
    return

  for i, (pid, step_idx, step) in enumerate(all_steps):
    pc = _player_color(pid)
    print(_color(f'  ┌─ Player {pid}, Step {step_idx} ', pc)
          + _color('─' * 50, pc))

    # State
    state_text = step.get('state_text', '')
    if state_text:
      print(_color('  │ State:', BOLD))
      for line in state_text.strip().split('\n'):
        print(f'  │   {line}')

    # Prompt (optional)
    if show_prompts:
      prompt = step.get('prompt', '')
      if prompt:
        print(_color('  │ Prompt:', DIM))
        for line in prompt.strip().split('\n'):
          print(f'  │   {DIM}{line}{RESET}')

    # LLM Response
    llm_resp = step.get('llm_response', '')
    if llm_resp:
      print(_color('  │ LLM Response: ', BOLD)
            + _color(repr(llm_resp.strip()), CYAN))

    # Parsed action
    game_action = step.get('game_action', '')
    action_id = step.get('action_id', '?')
    log_prob = step.get('log_prob', 0.0)
    print(f'  │ Action: {_color(game_action, GREEN)}'
          f'  (id={action_id}, log_prob={log_prob:.4f})')

    print(_color('  └' + '─' * 60, pc))
    print()

  # Final rewards summary
  print(_color('  Result:', BOLD))
  for p_data in players:
    pid = p_data['player_id']
    reward = p_data.get('reward', 0.0)
    color = GREEN if reward > 0 else (RED if reward < 0 else DIM)
    print(f'    Player {pid}: '
          + _color(f'{reward:.1f} reward', color))
  print()


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
  args = parse_args()

  episodes = load_episodes(args.log_path)
  if not episodes:
    print('No episodes found in log file.', file=sys.stderr)
    sys.exit(1)

  # Filter by episode numbers if specified.
  if args.episodes:
    target_eps = set(
        int(x.strip()) for x in args.episodes.split(',') if x.strip()
    )
    episodes = [e for e in episodes if e['episode'] in target_eps]

  # Show only last N.
  if args.last > 0:
    episodes = episodes[-args.last:]

  if not episodes:
    print('No episodes match the filter criteria.', file=sys.stderr)
    sys.exit(1)

  # Always show summary table.
  print_summary_table(episodes)

  # Show detailed view unless summary_only.
  if not args.summary_only:
    for ep in episodes:
      print_episode_detail(ep, show_prompts=args.show_prompts)


if __name__ == '__main__':
  main()

