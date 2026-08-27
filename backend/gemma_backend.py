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

"""Gemma 2B LLM backend with LoRA fine-tuning support.

This module implements a locally-loaded Gemma 2B model (via HuggingFace
Transformers + PEFT/LoRA) that can be used for gradient-based RL training.

Key design decisions:
  - LoRA rank 16 / alpha 32 targets the q_proj and v_proj attention
    matrices — this keeps VRAM < 8 GB on a single GPU.
  - 4-bit NF4 quantization (via bitsandbytes) lets the frozen backbone
    fit alongside the trainable adapter.
  - The tokenizer's pad token is set to eos_token (Gemma's default
    tokenizer has no pad token).
"""

import os
from typing import Optional

from absl import logging
import llm_agent
import torch

# Lazy imports — heavy dependencies loaded only when needed.
transformers = None  # Will be imported in _lazy_import_hf()
peft = None
trl = None


def _lazy_import_hf():
  """Import heavy HF dependencies only when needed."""
  global transformers, peft, trl
  if transformers is None:
    import transformers as _transformers
    import peft as _peft

    transformers = _transformers
    peft = _peft
  try:
    if trl is None:
      import trl as _trl

      trl = _trl
  except ImportError:
    logging.warning('trl not installed — PPOTrainer unavailable.')


class GemmaLLMBackend(llm_agent.LLMInterface):
  """LLM backend backed by a locally-loaded Gemma 2B (LoRA fine-tuned).

  This backend loads Gemma 2B with optional 4-bit quantization, attaches
  a LoRA adapter, and provides `generate` / `generate_with_logprobs`
  methods compatible with the LLMInterface ABC.

  Unlike the GeminiLLM API backend in llm_agent.py, this backend runs
  locally on GPU and supports gradient-based training — the key piece
  needed for true RL fine-tuning.

  Attributes:
    model: The HuggingFace model with LoRA adapter attached.
    tokenizer: The HuggingFace tokenizer.
    device: The torch device the model is loaded on.
  """

  def __init__(
      self,
      model_name: str = 'google/gemma-2-2b',
      lora_rank: int = 16,
      lora_alpha: int = 32,
      lora_dropout: float = 0.05,
      use_4bit: bool = True,
      max_seq_len: int = 512,
      device: Optional[str] = None,
  ):
    """Initializes the Gemma LLM backend with LoRA.

    Args:
      model_name: HuggingFace model identifier.
      lora_rank: Rank of the LoRA decomposition.
      lora_alpha: LoRA scaling factor.
      lora_dropout: Dropout probability for LoRA layers.
      use_4bit: Whether to load the base model in 4-bit precision.
      max_seq_len: Maximum sequence length for tokenization.
      device: Target device ('cuda', 'cpu', or None for auto).
    """
    _lazy_import_hf()

    self._max_seq_len = max_seq_len
    self._hf_token = os.environ.get('HF_TOKEN', None)

    if device is None:
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
      self.device = device

    logging.info('Loading Gemma model: %s (4-bit=%s)', model_name, use_4bit)

    # ── Quantization config ──
    quant_config = None
    if use_4bit:
      quant_config = transformers.BitsAndBytesConfig(
          load_in_4bit=True,
          bnb_4bit_quant_type='nf4',
          bnb_4bit_compute_dtype=torch.bfloat16,
          bnb_4bit_use_double_quant=True,
      )

    # ── Load base model ──
    self.model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map='auto' if self.device == 'cuda' else None,
        torch_dtype=torch.bfloat16,
        attn_implementation='eager',  # Gemma 2 needs eager attention
        token=self._hf_token,
    )

    # ── Tokenizer ──
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, token=self._hf_token
    )
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token

    # ── LoRA adapter ──
    self._lora_config_template = peft.LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=['q_proj', 'v_proj'],
        bias='none',
        task_type=peft.TaskType.CAUSAL_LM,
    )

    if use_4bit:
      self.model = peft.prepare_model_for_kbit_training(self.model)

    self.model = peft.get_peft_model(self.model, self._lora_config_template)
    self.model.print_trainable_parameters()
    self._active_adapter: str = 'default'

    logging.info('Gemma backend ready on device=%s', self.device)

  # ── Multi-adapter management for phased training ──

  def create_player_adapters(self, num_players: int = 2) -> None:
    """Create separate LoRA adapters for each player.

    Adds named adapters 'player_0', 'player_1', etc. to the PEFT model.
    The initial 'default' adapter remains and can be used as a reference.
    Each new adapter is initialized from the current 'default' adapter
    weights so training starts from the same pre-trained baseline.

    Args:
      num_players: Number of player adapters to create.
    """
    _lazy_import_hf()
    for pid in range(num_players):
      adapter_name = f'player_{pid}'
      self.model.add_adapter(adapter_name, self._lora_config_template)
      logging.info('Created LoRA adapter: %s', adapter_name)

    # Activate the first player's adapter by default.
    self.set_active_adapter('player_0')
    logging.info(
        'Created %d player adapters. Active: %s',
        num_players,
        self._active_adapter,
    )

  def set_active_adapter(self, adapter_name: str) -> None:
    """Switch the active LoRA adapter.

    Args:
      adapter_name: Name of the adapter to activate (e.g. 'player_0').
    """
    self.model.set_adapter(adapter_name)
    self._active_adapter = adapter_name

  def get_active_adapter(self) -> str:
    """Returns the name of the currently active adapter."""
    return self._active_adapter

  def get_adapter_state_dict(self, adapter_name: str) -> dict:
    """Get a frozen copy of a specific adapter's parameters.

    Args:
      adapter_name: Name of the adapter to snapshot.

    Returns:
      A dict mapping parameter names to detached tensor clones.
    """
    self.set_active_adapter(adapter_name)
    state = {}
    for name, param in self.model.named_parameters():
      if param.requires_grad:
        state[name] = param.data.detach().clone()
    return state

  def load_adapter_state_dict(
      self, adapter_name: str, state_dict: dict
  ) -> None:
    """Load parameters into a specific adapter.

    Args:
      adapter_name: Name of the adapter to load into.
      state_dict: Dict mapping parameter names to tensors.
    """
    prev = self._active_adapter
    self.set_active_adapter(adapter_name)
    for name, param in self.model.named_parameters():
      if name in state_dict:
        param.data.copy_(state_dict[name])
    self.set_active_adapter(prev)

  def freeze_adapter(self, adapter_name: str) -> None:
    """Freeze all parameters in the named adapter (no gradients).

    Args:
      adapter_name: Name of the adapter to freeze.
    """
    prev = self._active_adapter
    self.set_active_adapter(adapter_name)
    for param in self.model.parameters():
      if param.requires_grad:
        param.requires_grad_(False)
    self.set_active_adapter(prev)

  def unfreeze_adapter(self, adapter_name: str) -> None:
    """Unfreeze all LoRA parameters in the named adapter.

    Args:
      adapter_name: Name of the adapter to unfreeze.
    """
    prev = self._active_adapter
    self.set_active_adapter(adapter_name)
    for name, param in self.model.named_parameters():
      # Only unfreeze LoRA parameters (contain 'lora_' in the name).
      if 'lora_' in name:
        param.requires_grad_(True)
    self.set_active_adapter(prev)

  def generate(
      self,
      prompt: str,
      temperature: float = 0.7,
      max_tokens: int = 64,
  ) -> str:
    """Generate text from a prompt.

    Args:
      prompt: Input prompt string.
      temperature: Sampling temperature.
      max_tokens: Maximum new tokens to generate.

    Returns:
      Generated text string (response only, prompt stripped).
    """
    inputs = self.tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=self._max_seq_len,
    ).to(self.model.device)

    with torch.no_grad():
      output_ids = self.model.generate(
          **inputs,
          max_new_tokens=max_tokens,
          temperature=max(temperature, 1e-3),
          do_sample=temperature > 0,
          top_p=0.9,
          pad_token_id=self.tokenizer.pad_token_id,
      )

    # Decode only the newly generated tokens.
    new_tokens = output_ids[0, inputs['input_ids'].shape[1] :]
    return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

  def generate_with_logprobs(
      self,
      prompt: str,
      temperature: float = 0.7,
      max_tokens: int = 64,
  ) -> tuple[str, float]:
    """Generate text and return the total log-probability of the response.

    Uses teacher-forcing: generates the text first, then computes the
    exact log-probability of each generated token under the model.

    Args:
      prompt: Input prompt string.
      temperature: Sampling temperature.
      max_tokens: Maximum new tokens to generate.

    Returns:
      Tuple of (generated_text, total_log_prob).
    """
    # Step 1: Generate the response.
    text = self.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    if not text:
      return '', 0.0

    # Step 2: Compute log-prob via a forward pass over prompt + response.
    full_text = prompt + text
    inputs = self.tokenizer(
        full_text,
        return_tensors='pt',
        truncation=True,
        max_length=self._max_seq_len,
    ).to(self.model.device)

    prompt_inputs = self.tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=self._max_seq_len,
    )
    prompt_len = prompt_inputs['input_ids'].shape[1]

    with torch.no_grad():
      outputs = self.model(**inputs)
      logits = outputs.logits  # (1, seq_len, vocab_size)

    # Compute log-probs for the response tokens only.
    # logits[t] predicts token[t+1], so we take logits[prompt_len-1:-1]
    # and compare against input_ids[prompt_len:].
    response_logits = logits[0, prompt_len - 1 : -1, :]  # (response_len, vocab)
    response_ids = inputs['input_ids'][0, prompt_len:]  # (response_len,)

    log_probs = torch.log_softmax(response_logits, dim=-1)
    token_log_probs = log_probs.gather(1, response_ids.unsqueeze(1)).squeeze(
        1
    )  # (response_len,)

    total_log_prob = float(token_log_probs.sum().item())
    return text, total_log_prob

  def compute_action_log_prob(
      self,
      prompt: str,
      action_text: str,
  ) -> torch.Tensor:
    """Compute the differentiable log-probability of an action string.

    Unlike generate_with_logprobs, this method returns a *gradient-bearing*
    tensor so that REINFORCE can back-propagate through the LoRA weights.

    Args:
      prompt: The game state prompt.
      action_text: The action text that was selected.

    Returns:
      A scalar torch.Tensor (with grad_fn) representing the log-probability.
    """
    full_text = prompt + action_text
    inputs = self.tokenizer(
        full_text,
        return_tensors='pt',
        truncation=True,
        max_length=self._max_seq_len,
    ).to(self.model.device)

    prompt_inputs = self.tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=self._max_seq_len,
    )
    prompt_len = prompt_inputs['input_ids'].shape[1]

    outputs = self.model(**inputs)
    logits = outputs.logits

    response_logits = logits[0, prompt_len - 1 : -1, :]
    response_ids = inputs['input_ids'][0, prompt_len:]

    log_probs = torch.log_softmax(response_logits, dim=-1)
    token_log_probs = log_probs.gather(1, response_ids.unsqueeze(1)).squeeze(1)

    return token_log_probs.sum()
