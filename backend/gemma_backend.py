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

import concurrent.futures
import os
import shutil
from typing import Optional

from absl import logging
import llm_agent
import torch

try:
  from pyglib import gfile
except ImportError:
  try:
    from tensorflow.io import gfile
  except ImportError:
    gfile = None


def _stage_checkpoint_if_cns(checkpoint_path: str) -> str:
  """If checkpoint_path is in CNS, stages it to local /tmp to allow PEFT to load it."""
  if not checkpoint_path or not checkpoint_path.startswith('/cns/'):
    return checkpoint_path
  if gfile is None:
    return checkpoint_path
  local_dir = f'/tmp/cns_lora_ckpt_{abs(hash(checkpoint_path))}'
  os.makedirs(local_dir, exist_ok=True)
  list_fn = getattr(gfile, 'ListDir', getattr(gfile, 'listdir', None))
  copy_fn = getattr(gfile, 'Copy', getattr(gfile, 'copy', None))
  if list_fn and copy_fn:
    logging.info(
        'Staging CNS LoRA checkpoint %s to local %s', checkpoint_path, local_dir
    )
    try:
      for fname in list_fn(checkpoint_path):
        src = os.path.join(checkpoint_path, fname)
        dst = os.path.join(local_dir, fname)
        copy_fn(src, dst, overwrite=True)
      return local_dir
    except Exception as e:
      logging.exception('Failed to stage CNS checkpoint: %s', e)
  return checkpoint_path


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

  When multiple GPUs are available (torch.cuda.device_count() >= 2), a replica
  is loaded on cuda:1 for parallel multi-GPU prompt generation.

  Attributes:
    model: The HuggingFace model with LoRA adapter attached on primary device.
    tokenizer: The HuggingFace tokenizer.
    device: The torch device the primary model is loaded on.
  """

  def __init__(
      self,
      model_name: str = 'google/gemma-2-2b',
      lora_rank: int = 16,
      lora_alpha: int = 32,
      lora_dropout: float = 0.05,
      use_4bit: bool = True,
      max_seq_len: int = 2048,
      device: Optional[str] = None,
      lora_checkpoint: Optional[str] = None,
      base_lora_checkpoint: Optional[str] = None,
      fallback_lora_checkpoints: Optional[list[str]] = None,
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
      lora_checkpoint: Optional path to a pre-trained LoRA adapter to load (warm start or resume).
      base_lora_checkpoint: Optional path to a warm-start BC LoRA adapter to merge
        into the base model weights so that PEFT disable_adapter() uses the BC
        policy as the KL reference model rather than the un-finetuned base model.
      fallback_lora_checkpoints: Optional ordered list of fallback RL checkpoints
        to try if lora_checkpoint was interrupted mid-write on CNS.
    """
    _lazy_import_hf()

    self._max_seq_len = max_seq_len
    self._hf_token = os.environ.get('HF_TOKEN', None)

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if device is None:
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
      self.device = device

    self._replica_devices: list[str] = []
    if self.device == 'cuda' and num_gpus >= 2:
      primary_device_map = {'': 'cuda:0'}
      self._replica_devices = [f'cuda:{i}' for i in range(1, num_gpus)]
    elif self.device == 'cuda':
      primary_device_map = {'': 'cuda:0'} if num_gpus == 1 else 'auto'
    else:
      primary_device_map = None

    logging.info(
        'Loading Gemma model: %s (4-bit=%s, gpus=%d, replicas=%s)',
        model_name,
        use_4bit,
        num_gpus,
        self._replica_devices,
    )

    # ── Quantization config ──
    quant_config = None
    if use_4bit:
      try:
        import bitsandbytes  # noqa: F401

        quant_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
      except Exception as e:
        logging.warning(
            'bitsandbytes not available (%s); loading model in full bfloat16'
            ' precision instead of 4-bit.',
            e,
        )
        quant_config = None
        use_4bit = False

    # ── Load base model on primary device ──
    self.model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map=primary_device_map,
        torch_dtype=torch.bfloat16,
        attn_implementation='eager',  # Gemma 2 needs eager attention
        token=self._hf_token,
    )

    # ── Load replica models on secondary devices if available ──
    self._replica_models = []
    if self._replica_devices:
      for r_device in self._replica_devices:
        logging.info(
            'Detected %d GPUs. Initializing replica model on %s for concurrent prompt generation.',
            num_gpus,
            r_device,
        )
        replica = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map={'': r_device},
            torch_dtype=torch.bfloat16,
            attn_implementation='eager',
            token=self._hf_token,
        )
        self._replica_models.append(replica)

    # ── Tokenizer ──
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, token=self._hf_token
    )
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token
    self.tokenizer.padding_side = 'left'

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
      self._replica_models = [
          peft.prepare_model_for_kbit_training(m) for m in self._replica_models
      ]

    # If a warm-start BC checkpoint is provided (and not 4-bit), merge it into
    # the base model weights so TRL's disable_adapter() uses the BC policy as
    # the KL reference model rather than raw un-finetuned Gemma.
    effective_base_lora = base_lora_checkpoint
    if effective_base_lora is None and lora_checkpoint and not use_4bit:
      effective_base_lora = lora_checkpoint
      lora_checkpoint = None

    if effective_base_lora and not use_4bit:
      staged_base = _stage_checkpoint_if_cns(effective_base_lora)
      # Fail fast: continuing without the merge would silently run RL (and its
      # KL reference) from the un-finetuned base model instead of the BC policy.
      if not os.path.exists(staged_base):
        raise FileNotFoundError(
            f'Base LoRA checkpoint {effective_base_lora} could not be staged'
            f' locally ({staged_base} does not exist).'
        )
      logging.info(
          'Merging base LoRA checkpoint into backbone weights: %s (staged from %s)',
          staged_base,
          effective_base_lora,
      )
      peft_base = peft.PeftModel.from_pretrained(
          self.model, staged_base, is_trainable=False
      )
      self.model = peft_base.merge_and_unload()
      self._replica_models = [
          peft.PeftModel.from_pretrained(
              m, staged_base, is_trainable=False
          ).merge_and_unload()
          for m in self._replica_models
      ]
      logging.info(
          'Successfully merged base LoRA checkpoint into backbone weights.'
      )

    if lora_checkpoint == effective_base_lora and not use_4bit:
      lora_checkpoint = None

    candidate_ckpts = []
    if lora_checkpoint:
      candidate_ckpts.append(lora_checkpoint)
    if fallback_lora_checkpoints:
      for fb in fallback_lora_checkpoints:
        if fb and fb not in candidate_ckpts and fb != effective_base_lora:
          candidate_ckpts.append(fb)

    loaded_ckpt = False
    for ckpt in candidate_ckpts:
      staged_checkpoint = _stage_checkpoint_if_cns(ckpt)
      if os.path.exists(staged_checkpoint):
        try:
          logging.info(
              'Loading pre-trained LoRA adapter from: %s (staged from %s)',
              staged_checkpoint,
              ckpt,
          )
          self.model = peft.PeftModel.from_pretrained(
              self.model, staged_checkpoint, is_trainable=True
          )
          self._replica_models = [
              peft.PeftModel.from_pretrained(
                  m, staged_checkpoint, is_trainable=False
              )
              for m in self._replica_models
          ]
          loaded_ckpt = True
          break
        except Exception as e:
          logging.warning(
              'Failed to load LoRA checkpoint %s (staged: %s): %s. Trying next fallback.',
              ckpt,
              staged_checkpoint,
              e,
          )
      else:
        logging.warning(
            'LoRA checkpoint %s (staged: %s) does not exist!',
            ckpt,
            staged_checkpoint,
        )

    if not loaded_ckpt:
      logging.info('Initializing fresh trainable LoRA adapter on backbone.')
      self.model = peft.get_peft_model(self.model, self._lora_config_template)
      self._replica_models = [
          peft.get_peft_model(m, self._lora_config_template)
          for m in self._replica_models
      ]

    for m in self._replica_models:
      m.eval()
      for p in m.parameters():
        p.requires_grad_(False)
    self.sync_replica()

    self.model.print_trainable_parameters()
    self._active_adapter: str = 'default'

    logging.info(
        'Gemma backend ready on device=%s (replicas=%s)',
        self.device,
        self._replica_devices if self._replica_devices else 'None',
    )

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
      for m in self._replica_models:
        m.add_adapter(adapter_name, self._lora_config_template)
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
    for m in self._replica_models:
      m.set_adapter(adapter_name)
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
    for m in self._replica_models:
      for name, param in m.named_parameters():
        if name in state_dict:
          param.data.copy_(state_dict[name].to(param.device))
    self.set_active_adapter(prev)

  def sync_replica(self) -> None:
    """Synchronize LoRA adapter weights from primary model (cuda:0) to all replicas."""
    if not self._replica_models:
      return
    try:
      with torch.no_grad():
        primary_params = dict(self.model.named_parameters())
        for m in self._replica_models:
          for name, param in m.named_parameters():
            if name in primary_params and (
                'lora_' in name or primary_params[name].requires_grad
            ):
              param.data.copy_(primary_params[name].data.to(param.device))
          if hasattr(self.model, 'active_adapter') and hasattr(m, 'set_adapter'):
            if getattr(m, 'active_adapter', None) != getattr(
                self.model, 'active_adapter', None
            ):
              m.set_adapter(self.model.active_adapter)
      logging.info(
          'Synchronized LoRA adapter weights to %d replica models on %s',
          len(self._replica_models),
          self._replica_devices,
      )
    except Exception as e:
      logging.warning('Failed to sync weights to replica: %s', e)

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

  def _generate_on_model(
      self,
      model,
      prompts: list[str],
      temperature: float = 0.7,
      max_tokens: int = 64,
  ) -> list[str]:
    """Helper to run batched generation on a specific model replica."""
    if not prompts:
      return []

    device = model.device
    inputs = self.tokenizer(
        prompts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=self._max_seq_len,
    ).to(device)

    with torch.no_grad():
      output_ids = model.generate(
          **inputs,
          max_new_tokens=max_tokens,
          temperature=max(temperature, 1e-3) if temperature > 0 else 1.0,
          do_sample=temperature > 0,
          top_p=0.9 if temperature > 0 else 1.0,
          pad_token_id=self.tokenizer.pad_token_id,
      )

    input_len = inputs['input_ids'].shape[1]
    responses = []
    for i in range(len(prompts)):
      new_tokens = output_ids[i, input_len:]
      responses.append(
          self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
      )
    return responses

  def generate_batch(
      self,
      prompts: list[str],
      temperature: float = 0.7,
      max_tokens: int = 64,
  ) -> list[str]:
    """Generate text for a batch of prompts using GPU tensor batching.

    If multiple GPUs are available and len(prompts) >= 2, splits the prompts
    across primary and replica models for concurrent multi-GPU execution.

    Args:
      prompts: List of input prompt strings.
      temperature: Sampling temperature.
      max_tokens: Maximum new tokens to generate.

    Returns:
      List of generated text strings (response only, prompt stripped).
    """
    if not prompts:
      return []

    all_models = [self.model] + self._replica_models
    num_workers = min(len(all_models), len(prompts))
    if num_workers > 1:
      chunks = [[] for _ in range(num_workers)]
      for idx, prompt in enumerate(prompts):
        chunks[idx % num_workers].append((idx, prompt))

      results = [None] * len(prompts)
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=num_workers
      ) as executor:
        futures = []
        for i in range(num_workers):
          m = all_models[i]
          worker_prompts = [p for _, p in chunks[i]]
          futures.append(
              (
                  i,
                  executor.submit(
                      self._generate_on_model,
                      m,
                      worker_prompts,
                      temperature,
                      max_tokens,
                  ),
              )
          )
        for i, fut in futures:
          worker_res = fut.result()
          for (orig_idx, _), text in zip(chunks[i], worker_res):
            results[orig_idx] = text
      return results
    else:
      return self._generate_on_model(
          self.model, prompts, temperature, max_tokens
      )

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
