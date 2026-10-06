"""CPU-only end-to-end tests for preemption-safe resume in ``train_bc``.

A tiny randomly initialised Llama (1 layer, hidden size 16) with a word-level
tokenizer stands in for Gemma.  An uninterrupted reference run is compared
with runs that are "preempted" right after chosen resume checkpoints and then
restarted on the same --output_dir, as Borg does after an eviction: the
restarted runs must end with the same adapter weights, optimizer moments and
per-epoch metrics as the reference.
"""

import json
import os
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
import transformers

from google3.experimental.users.rfaulk.TeamGamesRL import train_bc

_ACTIONS = (
    'Play card 0',
    'Play card 1',
    'Discard card 0',
    'Discard card 1',
    'Hint Player 1 about Red cards',
    'Hint Player 1 about rank 1 cards',
)

# 24 train rows at batch size 2 with 2-step accumulation give 12 batches and
# 6 optimizer steps per epoch.  With --resume_every_steps=2 every epoch writes
# resume states at batches 4 and 8, at batch 12 (epoch done, before the
# evals), and at batch 0 of the next epoch (after the evals).
_NUM_TRAIN = 24
_NUM_VAL = 6
_EPOCHS = 2

# Tight enough to catch any replayed/skipped batch, missed optimizer or
# scheduler step, or different dropout mask (lr is 1e-2), loose enough to
# tolerate last-bit differences in CPU kernels.
_TOLERANCE = {'rtol': 1e-5, 'atol': 1e-7}


class _Preempted(BaseException):
  """Stands in for a Borg eviction.

  Derives from BaseException, like the SIGTERM-driven exit it models, so that
  train_bc's best-effort handling of failed resume saves cannot swallow it.
  """


def _preempt_after(kill_points):
  """Returns a write_resume_state stand-in that simulates evictions.

  Args:
    kill_points: (epoch, batches_done) resume positions.  Right after the
      state for each position is first written, the stand-in raises
      _Preempted (once per position).
  """
  pending = set(kill_points)
  real_write = train_bc.write_resume_state

  def write(state, output_dir):
    real_write(state, output_dir)
    position = (state['epoch'], state['batches_done'])
    if position in pending:
      pending.remove(position)
      raise _Preempted(position)

  return write


def _make_rows(n: int, offset: int) -> list[dict[str, str]]:
  legal = '\n'.join(f'- {a}' for a in _ACTIONS)
  rows = []
  for i in range(n):
    target = _ACTIONS[(i + offset) % len(_ACTIONS)]
    rows.append({
        'prompt': (
            f'Turn {i % 7} lives {(i + offset) % 3 + 1}\n'
            f'Legal actions:\n{legal}\nAction:'
        ),
        'completion': f'<think>\nBest action: {target}\n</think>\n{target}',
    })
  return rows


def _write_jsonl(path: str, rows: list[dict[str, str]]) -> str:
  with open(path, 'w') as f:
    for row in rows:
      f.write(json.dumps(row) + '\n')
  return path


def _save_tiny_model(model_dir: str, words: list[str]) -> None:
  """Saves a whitespace word-level tokenizer and a 1-layer Llama."""
  specials = ['[PAD]', '[UNK]', '[EOS]']
  vocab = {tok: i for i, tok in enumerate(specials + sorted(set(words)))}
  os.makedirs(model_dir, exist_ok=True)
  tokenizer_file = os.path.join(model_dir, 'tokenizer.json')
  with open(tokenizer_file, 'w') as f:
    json.dump(
        {
            'version': '1.0',
            'truncation': None,
            'padding': None,
            'added_tokens': [
                {
                    'id': vocab[tok],
                    'content': tok,
                    'single_word': False,
                    'lstrip': False,
                    'rstrip': False,
                    'normalized': False,
                    'special': True,
                }
                for tok in specials
            ],
            'normalizer': None,
            'pre_tokenizer': {'type': 'WhitespaceSplit'},
            'post_processor': None,
            'decoder': None,
            'model': {'type': 'WordLevel', 'vocab': vocab, 'unk_token': '[UNK]'},
        },
        f,
    )
  tokenizer = transformers.PreTrainedTokenizerFast(
      tokenizer_file=tokenizer_file,
      unk_token='[UNK]',
      pad_token='[PAD]',
      eos_token='[EOS]',
      # No token_type_ids: Llama's generate() rejects them.
      model_input_names=['input_ids', 'attention_mask'],
  )
  tokenizer.save_pretrained(model_dir)
  config = transformers.LlamaConfig(
      vocab_size=len(vocab),
      hidden_size=16,
      intermediate_size=32,
      num_hidden_layers=1,
      num_attention_heads=2,
      num_key_value_heads=1,
      max_position_embeddings=128,
      pad_token_id=vocab['[PAD]'],
      bos_token_id=None,
      eos_token_id=vocab['[EOS]'],
  )
  torch.manual_seed(0)
  transformers.LlamaForCausalLM(config).save_pretrained(model_dir)


def _load_metrics(output_dir: str) -> dict:
  with open(os.path.join(output_dir, 'bc_metrics.json')) as f:
    return json.load(f)


def _without_timing(action_metrics: dict) -> dict:
  return {k: v for k, v in action_metrics.items() if k != 'eval_seconds'}


class TrainBcResumeTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._root = tempfile.mkdtemp(dir=absltest.get_default_test_tmpdir())
    train_rows = _make_rows(_NUM_TRAIN, offset=0)
    val_rows = _make_rows(_NUM_VAL, offset=1)
    cls._train_file = _write_jsonl(
        os.path.join(cls._root, 'train.jsonl'), train_rows
    )
    cls._val_file = _write_jsonl(os.path.join(cls._root, 'val.jsonl'), val_rows)
    cls._model_dir = os.path.join(cls._root, 'model')
    _save_tiny_model(
        cls._model_dir,
        [
            word
            for row in train_rows + val_rows
            for word in (row['prompt'] + ' ' + row['completion']).split()
        ],
    )
    cls._reference_dir = os.path.join(cls._root, 'reference')
    train_bc.main(cls._argv(cls._reference_dir))

  @classmethod
  def _argv(cls, output_dir: str, **overrides) -> list[str]:
    flags = {
        'train_file': cls._train_file,
        'val_file': cls._val_file,
        'model_name': cls._model_dir,
        'output_dir': output_dir,
        'epochs': _EPOCHS,
        'batch_size': 2,
        'gradient_accumulation_steps': 2,
        'lr': 1e-2,
        'lora_rank': 4,
        'lora_alpha': 8,
        'max_seq_len': 64,
        'resume_every_steps': 2,
        'action_eval_samples': 4,
        'action_eval_max_new_tokens': 4,
        'action_eval_batch_size': 2,
    }
    flags.update(overrides)
    return ['train_bc'] + [f'--{k}={v}' for k, v in flags.items()]

  def _run_until_done(self, output_dir: str, kill_points) -> int:
    """Restarts train_bc after every simulated eviction; returns #attempts."""
    attempts = 0
    with mock.patch.object(
        train_bc, 'write_resume_state', _preempt_after(kill_points)
    ):
      while True:
        attempts += 1
        self.assertLessEqual(attempts, len(kill_points) + 1)
        try:
          train_bc.main(self._argv(output_dir))
          return attempts
        except _Preempted:
          pass

  def _assert_matches_reference(self, output_dir: str) -> None:
    want = train_bc.read_resume_state(self._reference_dir)
    got = train_bc.read_resume_state(output_dir)
    self.assertEqual((got['epoch'], got['batches_done']), (_EPOCHS, 0))
    self.assertSameElements(got['adapter'], want['adapter'])
    for name, tensor in want['adapter'].items():
      torch.testing.assert_close(got['adapter'][name], tensor, msg=name, **_TOLERANCE)
    self.assertSameElements(got['optimizer']['state'], want['optimizer']['state'])
    for idx, moments in want['optimizer']['state'].items():
      for key, value in moments.items():
        torch.testing.assert_close(
            got['optimizer']['state'][idx][key], value, msg=key, **_TOLERANCE
        )
    self.assertEqual(got['scheduler']['last_epoch'], want['scheduler']['last_epoch'])

    want_metrics = _load_metrics(self._reference_dir)
    got_metrics = _load_metrics(output_dir)
    self.assertEqual(got_metrics['init_val_loss'], want_metrics['init_val_loss'])
    self.assertEqual(
        [e['epoch'] for e in got_metrics['epochs']], list(range(1, _EPOCHS + 1))
    )
    for got_epoch, want_epoch in zip(got_metrics['epochs'], want_metrics['epochs']):
      self.assertAlmostEqual(got_epoch['train_loss'], want_epoch['train_loss'], places=5)
      self.assertAlmostEqual(got_epoch['val_loss'], want_epoch['val_loss'], places=5)
      self.assertEqual(
          _without_timing(got_epoch['action']), _without_timing(want_epoch['action'])
      )
    for sub_dir in ('epoch1_adapter', 'epoch2_adapter', 'best_adapter', 'final_adapter'):
      self.assertTrue(os.path.isdir(os.path.join(output_dir, sub_dir)), sub_dir)

  @parameterized.named_parameters(
      ('mid_epoch1', [(0, 4)]),
      ('epoch1_done_before_eval', [(0, 12)]),
      ('epoch1_done_after_eval', [(1, 0)]),
      ('mid_epoch2', [(1, 8)]),
      ('repeated_evictions', [(0, 8), (1, 4), (1, 12)]),
  )
  def test_preempted_run_matches_uninterrupted_run(self, kill_points):
    output_dir = self.create_tempdir().full_path
    attempts = self._run_until_done(output_dir, kill_points)
    self.assertEqual(attempts, len(kill_points) + 1)
    self._assert_matches_reference(output_dir)

  def test_comparison_is_sensitive_to_rng_restore(self):
    # Guards against _assert_matches_reference being vacuous: if the resumed
    # task does not restore the RNG, its dropout masks and weights diverge.
    output_dir = self.create_tempdir().full_path
    with mock.patch.object(torch, 'set_rng_state'):
      self._run_until_done(output_dir, [(0, 4)])
    want = train_bc.read_resume_state(self._reference_dir)['adapter']
    got = train_bc.read_resume_state(output_dir)['adapter']
    self.assertFalse(
        all(torch.allclose(got[k], want[k], **_TOLERANCE) for k in want)
    )

  def test_incompatible_resume_state_is_rejected(self):
    output_dir = self.create_tempdir().full_path
    with mock.patch.object(
        train_bc, 'write_resume_state', _preempt_after([(0, 4)])
    ):
      with self.assertRaises(_Preempted):
        train_bc.main(self._argv(output_dir))
    with self.assertRaisesRegex(ValueError, 'fresh --output_dir'):
      train_bc.main(self._argv(output_dir, batch_size=4))
    # --no-auto_resume starts over and replaces the stale state.
    train_bc.main(
        self._argv(output_dir, batch_size=4, epochs=1) + ['--no-auto_resume']
    )
    self.assertEqual(
        train_bc.read_resume_state(output_dir)['config']['batch_size'], 4
    )

  def test_resume_every_steps_zero_disables_resume_state(self):
    output_dir = self.create_tempdir().full_path
    train_bc.main(self._argv(output_dir, epochs=1, resume_every_steps=0))
    self.assertFalse(os.path.exists(os.path.join(output_dir, 'resume')))
    self.assertTrue(os.path.isdir(os.path.join(output_dir, 'final_adapter')))


class ResumeStateIoTest(absltest.TestCase):

  def test_round_trip_replaces_previous_state(self):
    output_dir = self.create_tempdir().full_path
    self.assertIsNone(train_bc.read_resume_state(output_dir))
    train_bc.write_resume_state({'epoch': 0, 'w': torch.ones(2)}, output_dir)
    train_bc.write_resume_state({'epoch': 1, 'w': torch.zeros(2)}, output_dir)
    state = train_bc.read_resume_state(output_dir)
    self.assertEqual(state['epoch'], 1)
    torch.testing.assert_close(state['w'], torch.zeros(2))
    self.assertEqual(
        os.listdir(os.path.join(output_dir, 'resume')), ['trainer_state.pt']
    )


class EpochShuffleSamplerTest(absltest.TestCase):

  def test_order_depends_only_on_seed_epoch_and_start(self):
    sampler = train_bc.EpochShuffleSampler(10, seed=3)
    torch.manual_seed(0)
    epoch0 = list(sampler)
    torch.manual_seed(1)  # The global RNG must not matter.
    self.assertEqual(list(sampler), epoch0)
    self.assertCountEqual(epoch0, range(10))
    self.assertEqual(list(train_bc.EpochShuffleSampler(10, seed=3)), epoch0)

    sampler.set_position(0, start=4)
    self.assertEqual(list(sampler), epoch0[4:])
    self.assertLen(sampler, 6)

    sampler.set_position(1)
    epoch1 = list(sampler)
    self.assertCountEqual(epoch1, range(10))
    self.assertNotEqual(epoch1, epoch0)

    sampler.set_position(1, start=10)
    self.assertEmpty(list(sampler))
    self.assertLen(sampler, 0)


if __name__ == '__main__':
  absltest.main()
