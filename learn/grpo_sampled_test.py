"""CPU-only tests for the GRPO sample-budget counters in ``learn.grpo_sampled``.

Covers ``_grpo_budget_metrics`` (per-pass and cumulative ``grpo/*`` keys, the
pass-0 zero anchor) and ``_load_grpo_budget_totals`` restoring the cumulative
counters of a resumed run from ``results/eval_metrics.csv`` in the format
written by ``RLTrainer._log_eval_metrics``.  Importing the module also smoke
tests that it still loads without an accelerator.
"""

import os
import sys

# Same bootstrap as trainer/gemma_rl_trainer.py: project-root imports
# (``env``, ``learn``, ``trainer``) and the ``open_spiel`` alias.
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
from learn import grpo_sampled  # pylint: disable=g-import-not-at-top

# pylint: disable=protected-access
_KEYS = grpo_sampled._GRPO_BUDGET_KEYS
_budget_metrics = grpo_sampled._grpo_budget_metrics
_load_totals = grpo_sampled._load_grpo_budget_totals
# pylint: enable=protected-access


class GrpoBudgetMetricsTest(absltest.TestCase):

  def test_emits_pass_and_total_keys(self):
    metrics = _budget_metrics(
        {
            'decision_points': 120,
            'rollouts': 960,
            'collected_decision_points': 1700,
        },
        {
            'decision_points': 370,
            'rollouts': 2960,
            'collected_decision_points': 5100,
        },
    )
    self.assertEqual(
        metrics,
        {
            'grpo/decision_points': 120,
            'grpo/decision_points_total': 370,
            'grpo/rollouts': 960,
            'grpo/rollouts_total': 2960,
            'grpo/collected_decision_points': 1700,
            'grpo/collected_decision_points_total': 5100,
        },
    )
    self.assertEqual(
        set(metrics), {f'grpo/{k}{s}' for k in _KEYS for s in ('', '_total')}
    )

  def test_missing_counts_are_zero(self):
    # The pass-0 baseline row uses this to anchor the curves at x = 0.
    metrics = _budget_metrics({}, {})
    self.assertLen(metrics, 2 * len(_KEYS))
    self.assertEqual(set(metrics.values()), {0})


class LoadGrpoBudgetTotalsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.results_dir = self.create_tempdir().full_path

  def _write_eval_csv(self, lines: list[str]) -> None:
    with open(os.path.join(self.results_dir, 'eval_metrics.csv'), 'w') as f:
      f.write('\n'.join(lines) + '\n')

  def test_restores_last_completed_pass(self):
    totals = ','.join(f'grpo/{k}_total' for k in _KEYS)
    self._write_eval_csv([
        f'episode,eval_full/mean_reward_p0,{totals},eval/mean_reward_p0',
        '0,1.000000,0.000000,0.000000,0.000000,',  # Baseline (pass 0).
        '50,2.000000,120.000000,960.000000,1700.000000,4.000000',
        '100,3.000000,250.000000,2000.000000,3400.000000,5.000000',
        '150,4.000000,370.000000,2960.000000,5100.000000,6.000000',
    ])
    self.assertEqual(
        _load_totals(self.results_dir, 100),
        {
            'decision_points': 250,
            'rollouts': 2000,
            'collected_decision_points': 3400,
        },
    )
    # Resuming at pass 4 (episode 150) picks the newest row; a rewind to
    # pass 1 (episode 50) ignores the later rows; pass 0 is the zero anchor.
    self.assertEqual(_load_totals(self.results_dir, 150)['decision_points'], 370)
    self.assertEqual(_load_totals(self.results_dir, 50)['decision_points'], 120)
    self.assertEqual(_load_totals(self.results_dir, 0), dict.fromkeys(_KEYS, 0))

  def test_legacy_csv_without_counters_restarts_at_zero(self):
    self._write_eval_csv([
        'episode,eval_full/mean_reward_p0',
        '0,1.000000',
        '50,2.000000',
    ])
    self.assertEqual(
        _load_totals(self.results_dir, 50), dict.fromkeys(_KEYS, 0)
    )

  def test_missing_csv_restarts_at_zero(self):
    self.assertEqual(
        _load_totals(self.results_dir, 50), dict.fromkeys(_KEYS, 0)
    )


if __name__ == '__main__':
  absltest.main()
