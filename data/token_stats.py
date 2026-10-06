"""Token-length statistics for BC JSONL datasets.

Reports prompt/completion token counts (using the same tokenization as
train_bc.py: completion = ' ' + completion.strip() + eos) so that RL
`max_completion_length` can be set from the data instead of guessed.

Example (CPU only):
  blaze run //experimental/users/rfaulk/TeamGamesRL:token_stats -- \
    --tokenizer=/tmp/gemma_tok \
    --files=/tmp/bc_hanabi_expertcot_1000g/val.jsonl,/tmp/bc_hanabi_reasoning_1000g/val.jsonl
"""

import json

from absl import app
from absl import flags
from transformers import AutoTokenizer

_TOKENIZER = flags.DEFINE_string(
    'tokenizer', None, 'Local dir with tokenizer.json / tokenizer.model.',
    required=True)
_FILES = flags.DEFINE_list(
    'files', None, 'Comma-separated JSONL files with prompt/completion.',
    required=True)
_LIMIT = flags.DEFINE_integer(
    'limit', 0, 'Only read the first N samples per file (0 = all).')
_OVER = flags.DEFINE_integer(
    'over', 256,
    'Also report the fraction of completions longer than this many tokens.')


def _percentile(sorted_vals, q):
  idx = min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))
  return sorted_vals[idx]


def _stats(vals):
  vals = sorted(vals)
  return {
      'n': len(vals),
      'mean': sum(vals) / len(vals),
      'p50': _percentile(vals, 0.5),
      'p90': _percentile(vals, 0.9),
      'p99': _percentile(vals, 0.99),
      'max': vals[-1],
  }


def _fmt(name, s):
  return (f'{name:<11s} n={s["n"]:<6d} mean={s["mean"]:7.1f} p50={s["p50"]:5d}'
          f' p90={s["p90"]:5d} p99={s["p99"]:5d} max={s["max"]:5d}')


def main(argv):
  del argv
  tok = AutoTokenizer.from_pretrained(_TOKENIZER.value)
  eos = 1 if tok.eos_token_id is not None else 0
  for path in _FILES.value:
    prompt_lens, comp_lens = [], []
    n_chars = 0
    longest = ('', 0)
    with open(path) as f:
      for i, line in enumerate(f):
        if _LIMIT.value and i >= _LIMIT.value:
          break
        line = line.strip()
        if not line:
          continue
        sample = json.loads(line)
        completion = ' ' + sample['completion'].strip()
        n_chars += len(completion)
        n_comp = len(tok.encode(completion, add_special_tokens=False)) + eos
        comp_lens.append(n_comp)
        prompt_lens.append(
            len(tok.encode(sample['prompt'], add_special_tokens=True)))
        if n_comp > longest[1]:
          longest = (sample['completion'], n_comp)
    over = sum(1 for n in comp_lens if n > _OVER.value) / len(comp_lens)
    print(f'== {path}')
    print('  ' + _fmt('prompt', _stats(prompt_lens)))
    print('  ' + _fmt('completion', _stats(comp_lens)))
    print(f'  completions > {_OVER.value} tokens: {over:.4%}')
    print(f'  chars/token (completion): {n_chars / sum(comp_lens):.2f}')
    print(f'  longest completion ({longest[1]} tokens):\n'
          + '\n'.join('    | ' + l for l in longest[0].splitlines()))


if __name__ == '__main__':
  app.run(main)
