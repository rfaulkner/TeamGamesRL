"""Supervised Fine-Tuning (Behavioral Cloning) for Hanabi on Gemma."""

import argparse
import json
import logging
import math
import os
import pathlib
import random
import shutil
import sys
import time

import peft
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
import transformers

try:
  from absl import app
  from absl import flags
except ImportError:
  app = None
  flags = None

try:
  from pyglib import gfile  # type: ignore
except ImportError:
  try:
    from tensorflow.io import gfile  # type: ignore
  except ImportError:
    gfile = None

# Ensure the repository root is on sys.path so imports resolve in all runtimes
_REPO_ROOT = pathlib.Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)


def resolve_file(path_str: str | pathlib.Path) -> str:
  path_str = str(path_str)
  if path_str.startswith('/cns/'):
    return path_str
  if os.path.exists(path_str):
    return path_str
  repo_path = _REPO_ROOT / path_str
  if repo_path.exists():
    return str(repo_path)
  if 'PYTHON_RUNFILES' in os.environ:
    rf_path = os.path.join(os.environ['PYTHON_RUNFILES'], 'google3', path_str)
    if os.path.exists(rf_path):
      return rf_path
    rf_path_direct = os.path.join(os.environ['PYTHON_RUNFILES'], path_str)
    if os.path.exists(rf_path_direct):
      return rf_path_direct
  return path_str


def open_file(path_str: str | pathlib.Path, mode: str = 'r'):
  path_str = str(path_str)
  if path_str.startswith('/cns/') and gfile is not None:
    if hasattr(gfile, 'Open'):
      return gfile.Open(path_str, mode)
    if hasattr(gfile, 'GFile'):
      return gfile.GFile(path_str, mode)
  return open(path_str, mode, encoding='utf-8' if 'b' not in mode else None)


class HanabiBCDataset(Dataset):
  """Dataset for supervised action prediction in Hanabi."""

  def __init__(
      self,
      jsonl_file: str | pathlib.Path,
      tokenizer: transformers.PreTrainedTokenizer,
      max_seq_len: int = 1024,
  ):
    resolved = resolve_file(jsonl_file)
    self.samples = []
    with open_file(resolved, 'r') as f:
      for line in f:
        line = line.strip()
        if line:
          self.samples.append(json.loads(line))

    self.tokenizer = tokenizer
    self.max_seq_len = max_seq_len
    logging.info(
        'Loaded %d samples from %s (resolved: %s)',
        len(self.samples),
        jsonl_file,
        resolved,
    )

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
    sample = self.samples[idx]
    prompt = sample['prompt']
    completion = ' ' + sample['completion'].strip()

    prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
    comp_ids = self.tokenizer.encode(completion, add_special_tokens=False)

    # Ensure eos_token is added to the completion
    if self.tokenizer.eos_token_id is not None:
      comp_ids.append(self.tokenizer.eos_token_id)

    full_ids = prompt_ids + comp_ids
    if len(full_ids) > self.max_seq_len:
      # Truncate prompt from the left if needed, keeping completion intact
      excess = len(full_ids) - self.max_seq_len
      prompt_ids = prompt_ids[excess:]
      full_ids = prompt_ids + comp_ids

    prompt_len = len(prompt_ids)
    labels = list(full_ids)
    # Mask out prompt tokens so loss is ONLY computed on completion
    for i in range(prompt_len):
      labels[i] = -100

    return {
        'input_ids': torch.tensor(full_ids, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
    }


def collate_fn(batch, pad_token_id: int):
  max_len = max(item['input_ids'].shape[0] for item in batch)
  input_ids_padded = []
  labels_padded = []
  attention_mask = []

  for item in batch:
    inp = item['input_ids']
    lbl = item['labels']
    pad_len = max_len - len(inp)

    padded_inp = torch.cat([inp, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
    padded_lbl = torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])
    mask = torch.cat([torch.ones_like(inp), torch.zeros(pad_len, dtype=torch.long)])

    input_ids_padded.append(padded_inp)
    labels_padded.append(padded_lbl)
    attention_mask.append(mask)

  return {
      'input_ids': torch.stack(input_ids_padded),
      'labels': torch.stack(labels_padded),
      'attention_mask': torch.stack(attention_mask),
  }


def evaluate(model, dataloader, device):
  model.eval()
  total_loss = 0.0
  total_tokens = 0
  with torch.no_grad():
    for batch in dataloader:
      input_ids = batch['input_ids'].to(device)
      labels = batch['labels'].to(device)
      attention_mask = batch['attention_mask'].to(device)

      outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
      loss = outputs.loss
      num_valid_tokens = (labels != -100).sum().item()

      total_loss += loss.item() * num_valid_tokens
      total_tokens += num_valid_tokens

  model.train()
  return total_loss / max(total_tokens, 1)


# ── Action accuracy ──────────────────────────────────────────────────────────
# Token loss is dominated by boilerplate (especially the templated <think>
# block), so the metric that matters for a cloned policy is: on held-out
# teacher decisions, how often does greedy decoding reproduce the teacher's
# action exactly?  Reported overall, per action type, and with a parse-health
# breakdown (is the prediction even a legal action?).


def extract_action_line(text: str) -> str:
  """Returns the action line of a completion (text after </think>, if any)."""
  if '</think>' in text:
    text = text.rsplit('</think>', 1)[1]
  for line in text.strip().splitlines():
    line = line.strip()
    if line:
      return line
  return ''


def normalize_action(text: str) -> str:
  return ' '.join(text.lower().replace('*', '').split()).rstrip('.')


def action_type(desc: str) -> str:
  lower = desc.strip().lower()
  if lower.startswith('play'):
    return 'play'
  if lower.startswith('discard'):
    return 'discard'
  if lower.startswith('hint') or 'reveal' in lower:
    return 'hint'
  return 'other'


def parse_legal_actions(prompt: str) -> list[str]:
  """Recovers the legal-action descriptions listed in a BC prompt."""
  marker = 'Legal actions:'
  if marker not in prompt:
    return []
  block = prompt.rsplit(marker, 1)[1]
  actions = []
  for line in block.splitlines():
    stripped = line.strip()
    if stripped.startswith('- '):
      actions.append(stripped[2:].strip())
    elif stripped.startswith('Action:'):
      break
  return actions


def evaluate_action_accuracy(
    model,
    tokenizer,
    samples: list[dict],
    device: str,
    max_new_tokens: int,
    batch_size: int,
    max_prompt_tokens: int,
) -> dict:
  """Greedy-decodes each prompt and scores exact match with the teacher action."""
  model.eval()
  prev_side = tokenizer.padding_side
  tokenizer.padding_side = 'left'
  stats = {
      'n': 0,
      'correct': 0,
      'legal': 0,
      'think_targets': 0,
      'think_unclosed': 0,
  }
  per_type = {t: {'n': 0, 'correct': 0} for t in ('play', 'discard', 'hint', 'other')}
  predicted_types = {t: 0 for t in ('play', 'discard', 'hint', 'other')}
  examples = []
  with torch.no_grad():
    for start in range(0, len(samples), batch_size):
      chunk = samples[start : start + batch_size]
      enc = tokenizer(
          [s['prompt'] for s in chunk],
          return_tensors='pt',
          padding=True,
          truncation=True,
          max_length=max_prompt_tokens,
          add_special_tokens=True,
      ).to(device)
      out = model.generate(
          **enc,
          max_new_tokens=max_new_tokens,
          do_sample=False,
          pad_token_id=tokenizer.pad_token_id,
          eos_token_id=tokenizer.eos_token_id,
      )
      gen = tokenizer.batch_decode(
          out[:, enc['input_ids'].shape[1] :], skip_special_tokens=True
      )
      for sample, text in zip(chunk, gen):
        target_line = extract_action_line(sample['completion'])
        pred_line = extract_action_line(text)
        t_type = action_type(target_line)
        p_type = action_type(pred_line)
        legal = {normalize_action(a) for a in parse_legal_actions(sample['prompt'])}
        is_correct = normalize_action(pred_line) == normalize_action(target_line)
        stats['n'] += 1
        stats['correct'] += int(is_correct)
        stats['legal'] += int(normalize_action(pred_line) in legal) if legal else 0
        per_type[t_type]['n'] += 1
        per_type[t_type]['correct'] += int(is_correct)
        predicted_types[p_type] += 1
        if '<think>' in sample['completion']:
          stats['think_targets'] += 1
          stats['think_unclosed'] += int('</think>' not in text)
        if len(examples) < 8:
          examples.append({'target': target_line, 'pred': pred_line, 'correct': is_correct})
  tokenizer.padding_side = prev_side
  model.train()

  n = max(stats['n'], 1)
  metrics = {
      'action_acc': stats['correct'] / n,
      'legal_rate': stats['legal'] / n,
      'n': stats['n'],
  }
  for t, d in per_type.items():
    metrics[f'acc_{t}'] = d['correct'] / d['n'] if d['n'] else None
    metrics[f'n_{t}'] = d['n']
    metrics[f'pred_frac_{t}'] = predicted_types[t] / n
  if stats['think_targets']:
    metrics['think_unclosed_rate'] = stats['think_unclosed'] / stats['think_targets']
  metrics['examples'] = examples
  return metrics


def format_action_metrics(m: dict) -> str:
  parts = [f"acc={m['action_acc']:.3f} (n={m['n']})", f"legal={m['legal_rate']:.3f}"]
  for t in ('play', 'discard', 'hint'):
    acc = m.get(f'acc_{t}')
    acc_s = f'{acc:.3f}' if acc is not None else 'n/a'
    parts.append(f"{t}: acc={acc_s} n={m.get(f'n_{t}', 0)} pred_frac={m.get(f'pred_frac_{t}', 0):.3f}")
  if 'think_unclosed_rate' in m:
    parts.append(f"think_unclosed={m['think_unclosed_rate']:.3f}")
  return ' | '.join(parts)


def stage_adapter_if_cns(adapter_path: str) -> str:
  """Copies a CNS adapter directory to /tmp so PEFT can load it."""
  if not adapter_path or not adapter_path.startswith('/cns/') or gfile is None:
    return adapter_path
  local_dir = pathlib.Path('/tmp') / f'bc_adapter_{abs(hash(adapter_path))}'
  local_dir.mkdir(parents=True, exist_ok=True)
  list_fn = getattr(gfile, 'ListDir', getattr(gfile, 'listdir', None))
  copy_fn = getattr(gfile, 'Copy', getattr(gfile, 'copy', None))
  for fname in list_fn(adapter_path):
    copy_fn(os.path.join(adapter_path, fname), str(local_dir / fname), overwrite=True)
  logging.info('Staged adapter %s -> %s', adapter_path, local_dir)
  return str(local_dir)


# ── Preemption-safe resume ───────────────────────────────────────────────────
# Single-GPU jobs share 8-GPU machines and get evicted by the borg-rescheduler
# to make room for whole-machine workloads (xid/295810817 was restarted 4
# times in 6h and, with no resume logic, re-trained from step 0 each time).
# The trainer therefore periodically writes everything needed to continue --
# adapter, optimizer, scheduler, RNG state and data position -- to
# <output_dir>/resume, and a restarted task picks up from there.  Data order
# comes from EpochShuffleSampler, so a resumed run sees exactly the batches an
# uninterrupted run would have seen.

_RESUME_STATE_FILE = os.path.join('resume', 'trainer_state.pt')


def _gfile_fn(*names: str):
  """Returns the first of `names` that the available gfile module provides."""
  for name in names:
    fn = getattr(gfile, name, None)
    if fn is not None:
      return fn
  raise AttributeError(f'gfile provides none of {names}')


def _uses_gfile(path: str) -> bool:
  return str(path).startswith('/cns/') and gfile is not None


def write_resume_state(state: dict, output_dir: str) -> None:
  """Atomically replaces <output_dir>/resume/trainer_state.pt with `state`."""
  final = os.path.join(output_dir, _RESUME_STATE_FILE)
  if not _uses_gfile(output_dir):
    os.makedirs(os.path.dirname(final), exist_ok=True)
    torch.save(state, final + '.tmp')
    os.replace(final + '.tmp', final)
    return
  local = f'/tmp/trainer_state_{os.getpid()}.pt'
  torch.save(state, local)
  try:
    resume_dir = os.path.dirname(final)
    if not _gfile_fn('Exists', 'exists')(resume_dir):
      _gfile_fn('MakeDirs', 'makedirs')(resume_dir)
    _gfile_fn('Copy', 'copy')(local, final + '.tmp', overwrite=True)
    _gfile_fn('Rename', 'rename')(final + '.tmp', final, overwrite=True)
  finally:
    os.remove(local)


def read_resume_state(output_dir: str) -> dict | None:
  """Returns the state last written by write_resume_state, or None."""
  final = os.path.join(output_dir, _RESUME_STATE_FILE)
  if not _uses_gfile(output_dir):
    if not os.path.exists(final):
      return None
    return torch.load(final, map_location='cpu', weights_only=True)
  if not _gfile_fn('Exists', 'exists')(final):
    return None
  local = f'/tmp/trainer_state_restore_{os.getpid()}.pt'
  _gfile_fn('Copy', 'copy')(final, local, overwrite=True)
  try:
    return torch.load(local, map_location='cpu', weights_only=True)
  finally:
    os.remove(local)


class EpochShuffleSampler(Sampler[int]):
  """Per-epoch shuffle determined by (seed, epoch) that can start mid-epoch.

  Unlike DataLoader(shuffle=True), the order does not depend on the global
  RNG, so after a restart `set_position(epoch, n_seen)` yields exactly the
  samples an uninterrupted run would still see in that epoch.
  """

  def __init__(self, num_samples: int, seed: int):
    self.num_samples = num_samples
    self.seed = seed
    self.epoch = 0
    self.start = 0

  def set_position(self, epoch: int, start: int = 0) -> None:
    self.epoch = epoch
    self.start = start

  def __iter__(self):
    generator = torch.Generator()
    generator.manual_seed(self.seed + self.epoch)
    order = torch.randperm(self.num_samples, generator=generator).tolist()
    return iter(order[self.start :])

  def __len__(self) -> int:
    return max(0, self.num_samples - self.start)


def main(argv=None):
  parser = argparse.ArgumentParser(description='Train BC LoRA model for Hanabi.')
  parser.add_argument('--train_file', type=str, required=True, help='Path to train.jsonl')
  parser.add_argument('--val_file', type=str, required=True, help='Path to val.jsonl')
  parser.add_argument('--model_name', type=str, default='google/gemma-3-12b-it', help='Base model name')
  parser.add_argument(
      '--use_4bit',
      action=argparse.BooleanOptionalAction,
      default=False,
      help='Use 4-bit quantization',
  )
  parser.add_argument('--lora_rank', type=int, default=16, help='LoRA rank')
  parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
  parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
  parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
  parser.add_argument('--batch_size', type=int, default=4, help='Batch size per step')
  parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
  parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
  parser.add_argument('--max_seq_len', type=int, default=2048, help='Max sequence length')
  parser.add_argument('--seed', type=int, default=42, help='Random seed')
  parser.add_argument('--output_dir', type=str, default='checkpoints/bc_gemma', help='Output checkpoint directory')
  parser.add_argument(
      '--action_eval_samples',
      type=int,
      default=500,
      help='Val decisions to greedy-decode for action accuracy each epoch (0 disables).',
  )
  parser.add_argument(
      '--action_eval_max_new_tokens',
      type=int,
      default=32,
      help=(
          'Generation budget for action accuracy. Use 512 for <think> data:'
          ' expert-CoT completions run up to ~350 tokens, and a 256 budget'
          ' cut off about two thirds of them (xid/295810817).'
      ),
  )
  parser.add_argument('--action_eval_batch_size', type=int, default=16, help='Generation batch size.')
  parser.add_argument(
      '--select_by',
      type=str,
      choices=['accuracy', 'val_loss'],
      default='accuracy',
      help='Metric used to pick best_adapter (falls back to val_loss if accuracy eval is off).',
  )
  parser.add_argument(
      '--save_every_epoch',
      action=argparse.BooleanOptionalAction,
      default=True,
      help='Also save epoch{N}_adapter after every epoch.',
  )
  parser.add_argument(
      '--init_adapter',
      type=str,
      default='',
      help='Existing LoRA adapter (local or CNS) to warm-start from, or to evaluate with --eval_only.',
  )
  parser.add_argument(
      '--eval_only',
      action=argparse.BooleanOptionalAction,
      default=False,
      help='Skip training: report val loss + action accuracy for --init_adapter (or the base model).',
  )
  parser.add_argument(
      '--resume_every_steps',
      type=int,
      default=500,
      help=(
          'Save a resume checkpoint to <output_dir>/resume every N optimizer'
          ' steps and at each epoch boundary, so a preempted task continues'
          ' where it stopped (0 disables saving).'
      ),
  )
  parser.add_argument(
      '--auto_resume',
      action=argparse.BooleanOptionalAction,
      default=True,
      help='Continue from <output_dir>/resume when it holds a resume checkpoint.',
  )
  raw_args = argv[1:] if argv is not None else None
  args, _ = parser.parse_known_args(raw_args)

  torch.manual_seed(args.seed)

  is_cns_output = str(args.output_dir).startswith('/cns/')
  if is_cns_output:
    local_output_path = pathlib.Path('/tmp') / f'checkpoints_{int(time.time())}'
  else:
    local_output_path = pathlib.Path(args.output_dir)
  local_output_path.mkdir(parents=True, exist_ok=True)

  def save_checkpoint(sub_name: str):
    save_local_dir = local_output_path / sub_name
    if save_local_dir.exists():
      shutil.rmtree(str(save_local_dir), ignore_errors=True)
    save_local_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_local_dir))
    tokenizer.save_pretrained(str(save_local_dir))
    logging.info('Saved %s locally to %s', sub_name, save_local_dir)
    if is_cns_output and gfile is not None:
      cns_dst_dir = os.path.join(args.output_dir, sub_name)
      logging.info('Uploading %s to CNS: %s', sub_name, cns_dst_dir)
      try:
        exists_fn = getattr(gfile, 'Exists', getattr(gfile, 'exists', None))
        makedirs_fn = getattr(
            gfile, 'MakeDirs', getattr(gfile, 'makedirs', None)
        )
        copy_fn = getattr(gfile, 'Copy', getattr(gfile, 'copy', None))
        if exists_fn and not exists_fn(cns_dst_dir):
          if makedirs_fn:
            makedirs_fn(cns_dst_dir)
        items = sorted(
            [p for p in save_local_dir.iterdir() if p.is_file()],
            key=lambda p: (0 if p.suffix == '.safetensors' else 1, p.name),
        )
        for item in items:
          if item.is_file():
            dst_file = os.path.join(cns_dst_dir, item.name)
            if copy_fn:
              try:
                copy_fn(str(item), dst_file, overwrite=True)
              except Exception:
                remove_fn = getattr(
                    gfile,
                    'Remove',
                    getattr(gfile, 'remove', getattr(gfile, 'Delete', None)),
                )
                if remove_fn and exists_fn and exists_fn(dst_file):
                  try:
                    remove_fn(dst_file)
                  except Exception:
                    pass
                copy_fn(str(item), dst_file)
        logging.info(
            'Successfully uploaded %s to CNS: %s', sub_name, cns_dst_dir
        )
        # Local /tmp on Borg is small (xid/295508481 died with ENOSPC while
        # saving epoch2_adapter): drop the local copy once it is on CNS.
        shutil.rmtree(str(save_local_dir), ignore_errors=True)
      except Exception as e:
        logging.exception('Failed to upload %s to CNS: %s', sub_name, e)

  def save_json(name: str, payload) -> None:
    """Writes a small JSON file locally and mirrors it to the CNS output dir."""
    local_file = local_output_path / name
    with open(local_file, 'w') as f:
      json.dump(payload, f, indent=2)
    if is_cns_output and gfile is not None:
      try:
        makedirs_fn = getattr(gfile, 'MakeDirs', getattr(gfile, 'makedirs', None))
        copy_fn = getattr(gfile, 'Copy', getattr(gfile, 'copy', None))
        if makedirs_fn:
          makedirs_fn(args.output_dir)
        copy_fn(str(local_file), os.path.join(args.output_dir, name), overwrite=True)
      except Exception as e:
        logging.warning('Failed to upload %s to CNS: %s', name, e)

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  logging.info('Training on device: %s', device)

  logging.info('Loading tokenizer: %s', args.model_name)
  tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = 'right'

  # Build Datasets
  train_dataset = HanabiBCDataset(args.train_file, tokenizer, args.max_seq_len)
  val_dataset = HanabiBCDataset(args.val_file, tokenizer, args.max_seq_len)

  # Seeded, position-aware shuffling (see EpochShuffleSampler) so a resumed
  # run replays the same batches.  The dedicated generator keeps DataLoader
  # iterator creation off the global RNG, which also drives dropout and is
  # saved/restored with the resume state.
  train_sampler = EpochShuffleSampler(len(train_dataset), args.seed)
  train_loader = DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      sampler=train_sampler,
      generator=torch.Generator(),
      collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
  )
  val_loader = DataLoader(
      val_dataset,
      batch_size=args.batch_size,
      shuffle=False,
      collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
  )

  bnb_config = None
  if args.use_4bit and torch.cuda.is_available():
    try:
      import bitsandbytes  # noqa: F401

      bnb_config = transformers.BitsAndBytesConfig(
          load_in_4bit=True,
          bnb_4bit_quant_type='nf4',
          bnb_4bit_compute_dtype=torch.bfloat16,
          bnb_4bit_use_double_quant=True,
      )
    except ImportError:
      logging.warning(
          'bitsandbytes not found; continuing with full bfloat16 precision'
          ' instead of 4-bit.'
      )
      args.use_4bit = False

  logging.info(
      'Loading base model: %s (4-bit=%s)', args.model_name, args.use_4bit
  )

  model = transformers.AutoModelForCausalLM.from_pretrained(
      args.model_name,
      quantization_config=bnb_config,
      device_map={'': 0} if torch.cuda.is_available() else None,
      torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
      attn_implementation='eager',
  )

  if args.use_4bit and torch.cuda.is_available():
    model = peft.prepare_model_for_kbit_training(model)

  lora_config = peft.LoraConfig(
      r=args.lora_rank,
      lora_alpha=args.lora_alpha,
      lora_dropout=args.lora_dropout,
      target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
      bias='none',
      task_type=peft.TaskType.CAUSAL_LM,
  )
  if args.init_adapter:
    staged = stage_adapter_if_cns(args.init_adapter)
    logging.info('Loading LoRA adapter from %s (trainable=%s)', staged, not args.eval_only)
    model = peft.PeftModel.from_pretrained(model, staged, is_trainable=not args.eval_only)
  else:
    model = peft.get_peft_model(model, lora_config)
  model.print_trainable_parameters()

  # Fixed, seed-chosen subset of val decisions for action accuracy.
  action_eval_samples = []
  if args.action_eval_samples > 0:
    n_eval = min(args.action_eval_samples, len(val_dataset.samples))
    idx = sorted(random.Random(args.seed).sample(range(len(val_dataset.samples)), n_eval))
    action_eval_samples = [val_dataset.samples[i] for i in idx]
  max_prompt_tokens = max(args.max_seq_len - args.action_eval_max_new_tokens, 256)
  # The tokenizer truncates on the right, which would cut off the legal-action
  # list at the end of the prompt and silently depress action accuracy.
  n_truncated = sum(
      len(tokenizer.encode(s['prompt'], add_special_tokens=True)) > max_prompt_tokens
      for s in action_eval_samples
  )
  if n_truncated:
    logging.warning(
        '%d/%d action-eval prompts exceed max_prompt_tokens=%d'
        ' (max_seq_len - action_eval_max_new_tokens) and will be'
        ' right-truncated, losing their legal-action list.',
        n_truncated,
        len(action_eval_samples),
        max_prompt_tokens,
    )

  def run_action_eval() -> dict | None:
    if not action_eval_samples:
      return None
    t_eval = time.time()
    m = evaluate_action_accuracy(
        model,
        tokenizer,
        action_eval_samples,
        device,
        max_new_tokens=args.action_eval_max_new_tokens,
        batch_size=args.action_eval_batch_size,
        max_prompt_tokens=max_prompt_tokens,
    )
    m['eval_seconds'] = time.time() - t_eval
    logging.info('Action accuracy: %s (%.0fs)', format_action_metrics(m), m['eval_seconds'])
    for ex in m['examples'][:4]:
      logging.info('  example: target=%r pred=%r correct=%s', ex['target'], ex['pred'], ex['correct'])
    return m

  if args.eval_only:
    val_loss = evaluate(model, val_loader, device)
    logging.info('[eval_only] Val Loss: %.4f', val_loss)
    action_metrics = run_action_eval()
    save_json(
        'eval_metrics.json',
        {
            'init_adapter': args.init_adapter,
            'val_file': args.val_file,
            'val_loss': val_loss,
            'action': action_metrics,
        },
    )
    logging.info('[eval_only] Done.')
    return

  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
  batches_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
  total_steps = (batches_per_epoch // args.gradient_accumulation_steps) * args.epochs
  scheduler = transformers.get_cosine_schedule_with_warmup(
      optimizer,
      num_warmup_steps=int(0.05 * total_steps),
      num_training_steps=total_steps,
  )

  logging.info('Starting SFT training: %d epochs, %d total optimization steps', args.epochs, total_steps)
  select_by_acc = args.select_by == 'accuracy' and bool(action_eval_samples)
  # A saved data position is only meaningful if these settings are unchanged.
  resume_config = {
      'train_file': args.train_file,
      'num_train_samples': len(train_dataset),
      'batch_size': args.batch_size,
      'gradient_accumulation_steps': args.gradient_accumulation_steps,
      'epochs': args.epochs,
      'seed': args.seed,
  }
  resumed = read_resume_state(args.output_dir) if args.auto_resume else None
  if resumed is not None:
    if resumed['config'] != resume_config:
      raise ValueError(
          f'Resume state in {args.output_dir} was written with'
          f' {resumed["config"]}, but this run uses {resume_config}. Use a'
          ' fresh --output_dir or --no-auto_resume.'
      )
    load_result = peft.set_peft_model_state_dict(model, resumed['adapter'])
    if load_result.unexpected_keys:
      raise ValueError(
          'Resume state has unexpected adapter keys:'
          f' {load_result.unexpected_keys[:5]}'
      )
    optimizer.load_state_dict(resumed['optimizer'])
    scheduler.load_state_dict(resumed['scheduler'])
    torch.set_rng_state(resumed['torch_rng'])
    if torch.cuda.is_available() and resumed['cuda_rng']:
      torch.cuda.set_rng_state_all(resumed['cuda_rng'])
    init_val_loss = resumed['init_val_loss']
    best_val_loss = resumed['best_val_loss']
    best_acc = resumed['best_acc']
    history = resumed['history']
    start_epoch = resumed['epoch']
    logging.info(
        'Resumed from %s at epoch %d/%d, batch %d/%d (skipping pre-SFT eval)',
        os.path.join(args.output_dir, _RESUME_STATE_FILE),
        start_epoch + 1,
        args.epochs,
        resumed['batches_done'],
        batches_per_epoch,
    )
  else:
    init_val_loss = evaluate(model, val_loader, device)
    logging.info('Initial (pre-SFT) Val Loss: %.4f', init_val_loss)
    best_val_loss = init_val_loss
    best_acc = -1.0
    history = []
    start_epoch = 0

  def save_resume_state(
      epoch: int,
      batches_done: int,
      step: int,
      running_loss: float,
      epoch_seconds: float,
  ) -> None:
    """Writes everything needed to continue after `batches_done` batches."""
    if args.resume_every_steps <= 0:
      return
    t_save = time.time()
    try:
      adapter = peft.get_peft_model_state_dict(model, save_embedding_layers=False)
      write_resume_state(
          {
              'config': resume_config,
              'epoch': epoch,
              'batches_done': batches_done,
              'step': step,
              'running_loss': running_loss,
              'epoch_seconds': epoch_seconds,
              'init_val_loss': init_val_loss,
              'best_val_loss': best_val_loss,
              'best_acc': best_acc,
              'history': history,
              'adapter': {k: v.detach().cpu() for k, v in adapter.items()},
              'optimizer': optimizer.state_dict(),
              'scheduler': scheduler.state_dict(),
              'torch_rng': torch.get_rng_state(),
              'cuda_rng': (
                  torch.cuda.get_rng_state_all()
                  if torch.cuda.is_available()
                  else []
              ),
          },
          args.output_dir,
      )
    except Exception:  # pylint: disable=broad-exception-caught
      # Best effort: a failed save must not kill the training it protects.
      logging.exception(
          'Failed to save resume state at epoch %d, batch %d',
          epoch + 1,
          batches_done,
      )
      return
    logging.info(
        'Saved resume state at epoch %d/%d, batch %d/%d (%.1fs)',
        epoch + 1,
        args.epochs,
        batches_done,
        batches_per_epoch,
        time.time() - t_save,
    )

  for epoch in range(start_epoch, args.epochs):
    model.train()
    if resumed is not None and epoch == start_epoch:
      start_batch = resumed['batches_done']
      running_loss = resumed['running_loss']
      step = resumed['step']
      prior_seconds = resumed['epoch_seconds']
    else:
      start_batch, running_loss, step, prior_seconds = 0, 0.0, 0, 0.0
    t0 = time.time()
    optimizer.zero_grad()
    train_sampler.set_position(epoch, start_batch * args.batch_size)

    for batch_idx, batch in enumerate(
        tqdm(
            train_loader,
            desc=f'Epoch {epoch+1}/{args.epochs}',
            initial=start_batch,
            total=batches_per_epoch,
        ),
        start=start_batch,
    ):
      input_ids = batch['input_ids'].to(device)
      labels = batch['labels'].to(device)
      attention_mask = batch['attention_mask'].to(device)

      outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
      loss = outputs.loss / args.gradient_accumulation_steps
      loss.backward()

      running_loss += outputs.loss.item()

      if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == batches_per_epoch:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        step += 1
        if step == 1 or step % 50 == 0:
          logging.info(
              'Epoch %d/%d | Step %d | Last Batch Loss: %.4f | Avg Train Loss: %.4f',
              epoch + 1,
              args.epochs,
              step,
              outputs.loss.item(),
              running_loss / (batch_idx + 1),
          )
        if (
            args.resume_every_steps > 0
            and step % args.resume_every_steps == 0
            and batch_idx + 1 < batches_per_epoch
        ):
          save_resume_state(
              epoch,
              batch_idx + 1,
              step,
              running_loss,
              prior_seconds + time.time() - t0,
          )

    # Protect the finished epoch before the slow evals: a restart from here
    # re-runs only the end-of-epoch evaluation instead of the whole epoch.
    save_resume_state(
        epoch,
        batches_per_epoch,
        step,
        running_loss,
        prior_seconds + time.time() - t0,
    )
    if args.save_every_epoch:
      save_checkpoint(f'epoch{epoch + 1}_adapter')

    train_loss = running_loss / batches_per_epoch
    val_loss = evaluate(model, val_loader, device)
    elapsed = prior_seconds + time.time() - t0

    logging.info(
        'Epoch %d/%d complete in %.1fs | Train Loss: %.4f | Val Loss: %.4f',
        epoch + 1,
        args.epochs,
        elapsed,
        train_loss,
        val_loss,
    )
    action_metrics = run_action_eval()
    history.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'epoch_seconds': elapsed,
        'action': action_metrics,
    })
    save_json('bc_metrics.json', {'args': vars(args), 'init_val_loss': init_val_loss, 'epochs': history})

    # Save checkpoint if best (by action accuracy when available, else val loss)
    if select_by_acc:
      acc = action_metrics['action_acc']
      if acc > best_acc:
        best_acc = acc
        save_checkpoint('best_adapter')
        logging.info('Saved new best adapter (action_acc=%.4f, val_loss=%.4f)', acc, val_loss)
    elif val_loss < best_val_loss:
      best_val_loss = val_loss
      save_checkpoint('best_adapter')
      logging.info('Saved new best adapter (val_loss=%.4f)', val_loss)
    save_resume_state(epoch + 1, 0, 0, 0.0, 0.0)

  # Also save final
  save_checkpoint('final_adapter')
  logging.info('Training complete. Checkpoints saved.')


if __name__ == '__main__':
  if app is not None and flags is not None:
    app.run(main, flags_parser=lambda argv: flags.FLAGS(argv, known_only=True))
  elif app is not None:
    app.run(main, flags_parser=lambda argv: argv)
  else:
    main()
