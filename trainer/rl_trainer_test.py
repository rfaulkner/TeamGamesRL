"""CPU-only tests for fixed-deal evaluation in ``trainer.rl_trainer``.

Covers:
  * Seeded Hanabi environments reproduce the same deal (paired evaluation).
  * ``RLTrainer._state_snapshot`` on Hanabi states and on bare objects.
  * ``RLTrainer.evaluate`` in its three flavours -- curriculum (LLM for
    ``[0, H)`` then bot), all-bot (``H=0``) and full self-play (``H=None``) --
    through both the batched and the sequential code paths, including the
    ``metric_prefix`` renaming and the ``eval_episodes.jsonl`` records.
  * The schema-tolerant ``eval_metrics.csv`` writer.

The LLM is replaced by ``llm_agent.MockLLM`` (random legal actions) so no
model weights or accelerators are needed.
"""

import csv
import json
import os
import sys
from typing import Any
from unittest import mock

# Same bootstrap as trainer/gemma_rl_trainer.py: project-root imports
# (``env``, ``trainer``, ``llm_agent``) and the ``open_spiel`` alias.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)
if 'open_spiel' not in sys.modules:
  try:
    import open_spiel  # pylint: disable=g-import-not-at-top,unused-import
  except ImportError:
    try:
      import third_party.open_spiel as _os  # pylint: disable=g-import-not-at-top
      sys.modules['open_spiel'] = _os
      import third_party.open_spiel.python as _osp  # pylint: disable=g-import-not-at-top
      sys.modules['open_spiel.python'] = _osp
    except ImportError:
      try:
        import google3.third_party.open_spiel as _os  # pylint: disable=g-import-not-at-top
        sys.modules['open_spiel'] = _os
        import google3.third_party.open_spiel.python as _osp  # pylint: disable=g-import-not-at-top
        sys.modules['open_spiel.python'] = _osp
      except ImportError:
        pass

from absl.testing import absltest  # pylint: disable=g-import-not-at-top
from env import game_config  # pylint: disable=g-import-not-at-top
from env import game_env  # pylint: disable=g-import-not-at-top
import llm_agent  # pylint: disable=g-import-not-at-top
import torch  # pylint: disable=g-import-not-at-top
from trainer import rl_trainer  # pylint: disable=g-import-not-at-top


class _FakeBackend(llm_agent.MockLLM):
  """MockLLM with the tiny ``model`` attribute ``RLTrainer`` expects."""

  def __init__(self, seed: int = 0):
    super().__init__(seed=seed)
    self.model = torch.nn.Linear(1, 1)


def _turn0_observations(env) -> tuple[str, str]:
  state = env._state  # pylint: disable=protected-access
  return state.observation_string(0), state.observation_string(1)


def _fixed_policy_rollout(env) -> tuple[list[int], int, int]:
  """Plays a deterministic (deal-dependent) policy; returns actions/score."""
  ts = env.reset()
  actions: list[int] = []
  while not ts.last():
    legal = env._state.legal_actions()  # pylint: disable=protected-access
    action = legal[len(actions) % len(legal)]
    actions.append(action)
    ts = env.step([action])
  state = env._state  # pylint: disable=protected-access
  return actions, state.score(), state.life_tokens()


class SeededEnvTest(absltest.TestCase):

  def test_same_seed_same_deal(self):
    cfg = game_config.HANABI_CONFIG
    env_a = game_env.create_env(cfg, seed=7)
    env_b = game_env.create_env(cfg, seed=7)
    self.assertEqual(env_a.game.seed, 7)
    env_a.reset()
    env_b.reset()
    self.assertEqual(_turn0_observations(env_a), _turn0_observations(env_b))

  def test_same_seed_same_draw_order(self):
    cfg = game_config.HANABI_CONFIG
    roll_a = _fixed_policy_rollout(game_env.create_env(cfg, seed=11))
    roll_b = _fixed_policy_rollout(game_env.create_env(cfg, seed=11))
    self.assertEqual(roll_a, roll_b)
    self.assertNotEmpty(roll_a[0])

  def test_different_seed_different_deal(self):
    cfg = game_config.HANABI_CONFIG
    env_a = game_env.create_env(cfg, seed=7)
    env_b = game_env.create_env(cfg, seed=8)
    env_a.reset()
    env_b.reset()
    self.assertNotEqual(
        _turn0_observations(env_a), _turn0_observations(env_b)
    )

  def test_unseeded_env_has_no_seed(self):
    env = game_env.create_env(game_config.HANABI_CONFIG)
    self.assertIsNone(env.game.seed)
    env.reset()  # Still playable.


class StateSnapshotTest(absltest.TestCase):

  def test_hanabi_initial_state(self):
    env = game_env.create_env(game_config.HANABI_CONFIG, seed=3)
    env.reset()
    snap = rl_trainer.RLTrainer._state_snapshot(env._state, 0)  # pylint: disable=protected-access
    self.assertEqual(
        snap,
        {
            'turn': 0.0,
            'score': 0.0,
            'lives': 3.0,
            'info_tokens': 8.0,
            'deck_size': 40.0,  # 50 cards - 2 players x 5 cards.
        },
    )

  def test_object_without_accessors(self):
    self.assertEqual(
        rl_trainer.RLTrainer._state_snapshot(object(), 5),  # pylint: disable=protected-access
        {'turn': 5.0},
    )


class EvaluateTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.output_dir = self.create_tempdir().full_path
    self.trainer = self._make_trainer(self.output_dir)

  def _make_trainer(self, output_dir: str, **kwargs) -> rl_trainer.RLTrainer:
    params = dict(
        game_name='hanabi',
        backend=_FakeBackend(seed=0),
        output_dir=output_dir,
        bot_type='safe_play',
        eval_batch_size=4,
        log_episodes_every=0,
        reasoning=False,
    )
    params.update(kwargs)
    return rl_trainer.RLTrainer(**params)

  def _records(self, tag: str) -> list[dict[str, Any]]:
    path = os.path.join(self.trainer.results_dir, 'eval_episodes.jsonl')
    with open(path) as f:
      recs = [json.loads(line) for line in f if line.strip()]
    return [r for r in recs if r['eval_tag'] == tag]

  def test_curriculum_eval_hands_off_at_horizon(self):
    m = self.trainer.evaluate(
        4, eval_llm_max_horizon=2, seed_base=123, metric_prefix='eval_cur'
    )
    self.assertTrue(all(k.startswith('eval_cur/') for k in m), m.keys())
    self.assertEqual(m['eval_cur/handoff_mean_turn'], 2.0)
    self.assertEqual(m['eval_cur/mean_llm_turns'], 2.0)
    self.assertGreater(m['eval_cur/mean_game_length'], 2.0)
    self.assertIn('eval_cur/handoff_mean_score', m)
    self.assertIn('eval_cur/handoff_mean_lives_lost', m)
    self.assertIn('eval_cur/handoff_mean_info_tokens', m)
    self.assertIn('eval_cur/mean_lives_lost', m)
    self.assertIn('eval_cur/bomb_out_rate', m)
    self.assertIn('eval_cur/std_reward_p0', m)
    # The LLM generated 2 moves per episode, so parse stats exist.
    self.assertIn('eval_cur/parse_fail_rate', m)

    recs = self._records('eval_cur')
    self.assertEqual([r['seed'] for r in recs], [123, 124, 125, 126])
    self.assertEqual({r['horizon'] for r in recs}, {2})
    for r in recs:
      self.assertEqual(r['handoff']['turn'], 2.0)
      self.assertEqual(r['stats']['llm_turns'], 2.0)
      llm_steps = [
          s for p in r['players'] for s in p['steps'] if s['prompt']
      ]
      self.assertLen(llm_steps, 2)

  def test_all_bot_baseline(self):
    m = self.trainer.evaluate(
        4, eval_llm_max_horizon=0, seed_base=123, metric_prefix='eval_bot'
    )
    self.assertEqual(m['eval_bot/mean_llm_turns'], 0.0)
    self.assertEqual(m['eval_bot/handoff_mean_turn'], 0.0)
    self.assertEqual(m['eval_bot/handoff_mean_score'], 0.0)
    # No LLM generations -> no parse statistics in this row.
    self.assertNotIn('eval_bot/parse_fail_rate', m)
    self.assertNotIn('eval_bot/mean_completion_tokens', m)
    recs = self._records('eval_bot')
    self.assertEqual({r['horizon'] for r in recs}, {0})
    for r in recs:
      self.assertTrue(
          all(not s['prompt'] for p in r['players'] for s in p['steps'])
      )

  def test_full_selfplay_eval(self):
    m = self.trainer.evaluate(4, seed_base=123, metric_prefix='eval_full')
    self.assertEqual(
        m['eval_full/mean_llm_turns'], m['eval_full/mean_game_length']
    )
    self.assertNotIn('eval_full/handoff_mean_turn', m)
    self.assertIn('eval_full/parse_fail_rate', m)
    recs = self._records('eval_full')
    self.assertEqual([r['seed'] for r in recs], [123, 124, 125, 126])
    self.assertEqual({r['horizon'] for r in recs}, {None})
    for r in recs:
      self.assertIsNone(r['handoff'])

  def test_same_seed_base_gives_paired_deals_across_calls(self):
    self.trainer.evaluate(2, seed_base=123, metric_prefix='eval_full')
    self.trainer.evaluate(
        2, eval_llm_max_horizon=2, seed_base=123, metric_prefix='eval_cur'
    )
    full = {r['seed']: r for r in self._records('eval_full')}
    cur = {r['seed']: r for r in self._records('eval_cur')}
    self.assertEqual(set(full), {123, 124})
    self.assertEqual(set(cur), {123, 124})
    for seed in (123, 124):
      # Turn-0 state text depends only on the deal: identical across calls.
      self.assertEqual(
          full[seed]['players'][0]['steps'][0]['state_text'],
          cur[seed]['players'][0]['steps'][0]['state_text'],
      )
    self.assertNotEqual(
        full[123]['players'][0]['steps'][0]['state_text'],
        full[124]['players'][0]['steps'][0]['state_text'],
    )

  def test_sequential_path_matches_batched_contract(self):
    self.trainer.eval_batch_size = 1
    m = self.trainer.evaluate(
        2, eval_llm_max_horizon=2, seed_base=500, metric_prefix='eval_cur'
    )
    self.assertEqual(m['eval_cur/handoff_mean_turn'], 2.0)
    self.assertEqual(m['eval_cur/mean_llm_turns'], 2.0)
    recs = self._records('eval_cur')
    self.assertEqual([r['seed'] for r in recs], [500, 501])
    m_bot = self.trainer.evaluate(
        2, eval_llm_max_horizon=0, seed_base=500, metric_prefix='eval_bot'
    )
    self.assertEqual(m_bot['eval_bot/mean_llm_turns'], 0.0)
    self.assertNotIn('eval_bot/parse_fail_rate', m_bot)

  def test_eval_generation_budget(self):
    # <think> policies need their full completion budget at eval time (expert
    # CoT runs to ~350 tokens); truncated output becomes a random fallback.
    cases = (
        (dict(reasoning=False), 64),
        (dict(reasoning=True), 256),
        (dict(reasoning=True, eval_max_tokens=512), 512),
    )
    for kwargs, want in cases:
      for eval_batch_size in (4, 1):  # Batched and sequential paths.
        with self.subTest(eval_batch_size=eval_batch_size, **kwargs):
          trainer = self._make_trainer(
              self.create_tempdir().full_path,
              eval_batch_size=eval_batch_size,
              **kwargs,
          )
          with mock.patch.object(
              trainer.backend, 'generate', wraps=trainer.backend.generate
          ) as generate:
            trainer.evaluate(2, eval_llm_max_horizon=2, seed_base=7)
          self.assertNotEmpty(generate.call_args_list)
          self.assertEqual(
              {c.kwargs['max_tokens'] for c in generate.call_args_list}, {want}
          )

  def test_default_prefix_and_unseeded_are_backward_compatible(self):
    m = self.trainer.evaluate(2, eval_llm_max_horizon=1)
    self.assertTrue(all(k.startswith('eval/') for k in m), m.keys())
    self.assertIn('eval/mean_reward_p0', m)
    self.assertIn('eval/win_rate_p0', m)
    recs = self._records('eval')
    self.assertEqual([r['seed'] for r in recs], [None, None])

  def test_eval_csv_is_schema_tolerant(self):
    path = self.trainer._eval_csv_path  # pylint: disable=protected-access
    self.trainer._log_eval_metrics(  # pylint: disable=protected-access
        0, {'eval_full/mean_reward_p0': 1.0, 'eval_bot/mean_reward_p0': 9.0}
    )
    self.trainer._log_eval_metrics(  # pylint: disable=protected-access
        10, {'eval/mean_reward_p0': 4.0, 'eval_full/mean_reward_p0': 2.0}
    )
    with open(path) as f:
      header = f.readline().strip().split(',')
      f.seek(0)
      rows = list(csv.DictReader(f))
    self.assertEqual(
        header,
        [
            'episode',
            'eval_bot/mean_reward_p0',
            'eval_full/mean_reward_p0',
            'eval/mean_reward_p0',
        ],
    )
    self.assertLen(rows, 2)
    self.assertEqual(rows[0]['episode'], '0')
    self.assertEqual(float(rows[0]['eval_bot/mean_reward_p0']), 9.0)
    self.assertEqual(rows[0]['eval/mean_reward_p0'], '')  # Padded old row.
    self.assertEqual(rows[1]['eval_bot/mean_reward_p0'], '')  # Missing cell.
    self.assertEqual(float(rows[1]['eval_full/mean_reward_p0']), 2.0)
    self.assertEqual(float(rows[1]['eval/mean_reward_p0']), 4.0)

    # A resumed run (new trainer, same output dir) keeps the schema.
    resumed = self._make_trainer(self.output_dir)
    self.assertEqual(resumed._eval_csv_columns, header[1:])  # pylint: disable=protected-access
    resumed._log_eval_metrics(20, {'eval/mean_reward_p0': 5.0})  # pylint: disable=protected-access
    with open(path) as f:
      lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    self.assertLen(lines, 4)
    self.assertTrue(all(ln.count(',') == 3 for ln in lines), lines)

  def test_read_eval_csv_row_restores_latest_consistent_snapshot(self):
    path = self.trainer._eval_csv_path  # pylint: disable=protected-access
    keys = ['grpo/decision_points_total', 'grpo/rollouts_total']
    log = self.trainer._log_eval_metrics  # pylint: disable=protected-access
    # Pass-0 baseline (zero anchor), then two passes; the header is widened
    # by the pass-1 row (new eval/* column) and pass 2 lacks one of the keys.
    log(0, {'eval_full/mean_reward_p0': 1.0} | dict.fromkeys(keys, 0))
    log(50, {'eval/mean_reward_p0': 4.0, keys[0]: 120, keys[1]: 960})
    log(100, {'eval/mean_reward_p0': 5.0, keys[0]: 250})
    log(150, {'eval/mean_reward_p0': 6.0, keys[0]: 370, keys[1]: 2960})

    read = rl_trainer.read_eval_csv_row
    # Latest row with *all* keys present wins; pass 2 (partial) is skipped.
    self.assertEqual(read(path, keys), {keys[0]: 370.0, keys[1]: 2960.0})
    self.assertEqual(read(path, keys, 100), {keys[0]: 120.0, keys[1]: 960.0})
    self.assertEqual(read(path, keys, 50), {keys[0]: 120.0, keys[1]: 960.0})
    self.assertEqual(read(path, keys, 0), {keys[0]: 0.0, keys[1]: 0.0})
    # A single key is found on every row that reports it.
    self.assertEqual(read(path, [keys[0]], 100), {keys[0]: 250.0})
    # Unknown column, missing file and no keys -> empty, never raises.
    self.assertEqual(read(path, ['grpo/nope_total']), {})
    self.assertEqual(read(os.path.join(self.output_dir, 'nope.csv'), keys), {})
    self.assertEqual(read(path, []), {})

  def test_eval_sink_receives_metrics_and_is_closed_once(self):
    calls: list[tuple[int, dict[str, float]]] = []
    closes: list[int] = []

    class _Sink:

      def log(self, ep, metrics):
        calls.append((ep, dict(metrics)))

      def close(self):
        closes.append(1)

    trainer = self._make_trainer(self.output_dir, eval_sink=_Sink())
    trainer._log_eval_metrics(50, {'eval/mean_reward_p0': 4.0})  # pylint: disable=protected-access
    self.assertEqual(calls, [(50, {'eval/mean_reward_p0': 4.0})])
    trainer._close_eval_sink()  # pylint: disable=protected-access
    trainer._close_eval_sink()  # pylint: disable=protected-access
    self.assertEqual(closes, [1])
    # After close the sink is detached: further evals only go to the CSV.
    trainer._log_eval_metrics(100, {'eval/mean_reward_p0': 5.0})  # pylint: disable=protected-access
    self.assertLen(calls, 1)

  def test_failing_eval_sink_does_not_break_logging(self):

    class _BrokenSink:

      def log(self, ep, metrics):
        raise RuntimeError('S2 down')

      def close(self):
        raise RuntimeError('S2 down')

    trainer = self._make_trainer(self.output_dir, eval_sink=_BrokenSink())
    trainer._log_eval_metrics(50, {'eval/mean_reward_p0': 4.0})  # pylint: disable=protected-access
    trainer._close_eval_sink()  # pylint: disable=protected-access
    with open(trainer._eval_csv_path) as f:  # pylint: disable=protected-access
      rows = list(csv.DictReader(f))
    self.assertLen(rows, 1)
    self.assertEqual(float(rows[0]['eval/mean_reward_p0']), 4.0)


if __name__ == '__main__':
  absltest.main()
