# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Supervised Fine-Tuning (Behavioral Cloning) for Hanabi on Gemma."""

import argparse
import json
import logging
import math
import os
import pathlib
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import transformers
import peft

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)


class HanabiBCDataset(Dataset):
  """Dataset for supervised action prediction in Hanabi."""

  def __init__(
      self,
      jsonl_file: str | pathlib.Path,
      tokenizer: transformers.PreTrainedTokenizer,
      max_seq_len: int = 1024,
  ):
    self.samples = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
      for line in f:
        line = line.strip()
        if line:
          self.samples.append(json.loads(line))

    self.tokenizer = tokenizer
    self.max_seq_len = max_seq_len
    logging.info('Loaded %d samples from %s', len(self.samples), jsonl_file)

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


def main():
  parser = argparse.ArgumentParser(description='Train BC LoRA model for Hanabi.')
  parser.add_argument('--train_file', type=str, required=True, help='Path to train.jsonl')
  parser.add_argument('--val_file', type=str, required=True, help='Path to val.jsonl')
  parser.add_argument('--model_name', type=str, default='google/gemma-3-12b-it', help='Base model name')
  parser.add_argument('--use_4bit', action='store_true', default=True, help='Use 4-bit quantization')
  parser.add_argument('--lora_rank', type=int, default=16, help='LoRA rank')
  parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
  parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
  parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
  parser.add_argument('--batch_size', type=int, default=4, help='Batch size per step')
  parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
  parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
  parser.add_argument('--max_seq_len', type=int, default=1024, help='Max sequence length')
  parser.add_argument('--output_dir', type=str, default='checkpoints/bc_gemma', help='Output checkpoint directory')
  parser.add_argument('--seed', type=int, default=42, help='Random seed')
  args = parser.parse_args()

  torch.manual_seed(args.seed)
  output_path = pathlib.Path(args.output_dir)
  output_path.mkdir(parents=True, exist_ok=True)

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

  train_loader = DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      shuffle=True,
      collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
  )
  val_loader = DataLoader(
      val_dataset,
      batch_size=args.batch_size,
      shuffle=False,
      collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
  )

  logging.info('Loading base model: %s (4-bit=%s)', args.model_name, args.use_4bit)
  bnb_config = None
  if args.use_4bit and torch.cuda.is_available():
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

  model = transformers.AutoModelForCausalLM.from_pretrained(
      args.model_name,
      quantization_config=bnb_config,
      device_map='auto' if torch.cuda.is_available() else None,
      torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
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
  model = peft.get_peft_model(model, lora_config)
  model.print_trainable_parameters()

  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
  total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.epochs
  scheduler = transformers.get_cosine_schedule_with_warmup(
      optimizer,
      num_warmup_steps=int(0.05 * total_steps),
      num_training_steps=total_steps,
  )

  logging.info('Starting SFT training: %d epochs, %d total optimization steps', args.epochs, total_steps)
  best_val_loss = float('inf')

  for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    step = 0
    t0 = time.time()
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')):
      input_ids = batch['input_ids'].to(device)
      labels = batch['labels'].to(device)
      attention_mask = batch['attention_mask'].to(device)

      outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
      loss = outputs.loss / args.gradient_accumulation_steps
      loss.backward()

      running_loss += outputs.loss.item()

      if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        step += 1

    train_loss = running_loss / len(train_loader)
    val_loss = evaluate(model, val_loader, device)
    elapsed = time.time() - t0

    logging.info(
        'Epoch %d/%d complete in %.1fs | Train Loss: %.4f | Val Loss: %.4f',
        epoch + 1,
        args.epochs,
        elapsed,
        train_loss,
        val_loss,
    )

    # Save checkpoint if best
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_dir = output_path / 'best_adapter'
      model.save_pretrained(best_dir)
      tokenizer.save_pretrained(best_dir)
      logging.info('Saved new best adapter to %s (val_loss=%.4f)', best_dir, val_loss)

  # Also save final
  final_dir = output_path / 'final_adapter'
  model.save_pretrained(final_dir)
  tokenizer.save_pretrained(final_dir)
  logging.info('Saved final adapter to %s', final_dir)


if __name__ == '__main__':
  main()
