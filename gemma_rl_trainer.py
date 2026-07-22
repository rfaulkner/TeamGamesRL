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

"""Gemma 2B RL trainer for multi-agent OpenSpiel team games.

This module integrates a locally-loaded Gemma 2B model (via HuggingFace
Transformers + PEFT/LoRA) with the OpenSpiel game environment to perform
gradient-based REINFORCE training on cooperative and competitive games.

Architecture overview:
  1. Load Gemma 2B with 4-bit quantization and attach a LoRA adapter so
     that only a small number of parameters are trainable (~0.5% of 2B).
  2. Wrap the model as a GemmaLLMBackend that implements the LLMInterface
     from llm_agent.py — it produces action distributions over legal game
     actions *and* returns per-token log-probabilities needed for
     policy-gradient updates.
  3. Run the existing RLTrainer episode loop (from train.py), but after
     each episode compute the REINFORCE loss and back-propagate through
     the LoRA parameters.
  4. Periodically evaluate and checkpoint.

Key design decisions:
  - LoRA rank 16 / alpha 32 targets the q_proj and v_proj attention
    matrices — this keeps VRAM < 8 GB on a single GPU.
  - 4-bit NF4 quantization (via bitsandbytes) lets the frozen backbone
    fit alongside the trainable adapter.
  - The tokenizer's pad token is set to eos_token (Gemma's default
    tokenizer has no pad token).

Usage:
  python gemma_rl_trainer.py --game=tiny_hanabi --num_episodes=500
  python gemma_rl_trainer.py --game=hanabi --lora_rank=32 --lr=5e-5
"""

import dataclasses
import json
import os
import time
from typing import Optional

from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch

import llm_agent
import state_renderers

# Lazy imports — heavy dependencies loaded only when needed.
transformers = None  # Will be imported in _lazy_import_hf()
peft = None
trl = None


# ============================================================================
# Flags
# ============================================================================

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    'game', 'tiny_hanabi', ['negotiation', 'hanabi', 'tiny_hanabi'],
    'Name of the OpenSpiel game to train on.')
flags.DEFINE_integer(
    'num_episodes', 500,
    'Total number of training episodes to run.')
flags.DEFINE_integer(
    'eval_every', 50,
    'Run evaluation every this many episodes.')
flags.DEFINE_integer(
    'num_eval_episodes', 10,
    'Number of episodes per evaluation round.')
flags.DEFINE_float(
    'temperature', 0.8,
    'Sampling temperature for LLM action selection.')
flags.DEFINE_float(
    'lr', 1e-4,
    'Learning rate for the LoRA adapter.')
flags.DEFINE_integer(
    'lora_rank', 16,
    'LoRA adapter rank.')
flags.DEFINE_integer(
    'lora_alpha', 32,
    'LoRA scaling alpha.')
flags.DEFINE_float(
    'lora_dropout', 0.05,
    'LoRA dropout probability.')
flags.DEFINE_string(
    'model_name', 'google/gemma-2-2b',
    'HuggingFace model ID for Gemma 2B.')
flags.DEFINE_bool(
    'use_4bit', True,
    'Use 4-bit NF4 quantization for the base model.')
flags.DEFINE_string(
    'output_dir', '/tmp/teamgamesrl',
    'Directory for checkpoints, logs, and metrics.')
flags.DEFINE_integer(
    'log_every', 10,
    'Log training metrics every this many episodes.')
flags.DEFINE_integer(
    'checkpoint_every', 100,
    'Save a LoRA checkpoint every this many episodes.')
flags.DEFINE_integer(
    'seed', 42,
    'Random seed for reproducibility.')
flags.DEFINE_integer(
    'max_seq_len', 512,
    'Maximum sequence length for the model.')
flags.DEFINE_float(
    'max_grad_norm', 1.0,
    'Maximum gradient norm for clipping.')
flags.DEFINE_bool(
    'use_wandb', False,
    'Enable Weights & Biases logging.')
flags.DEFINE_string(
    'wandb_project', 'TeamGamesRL',
    'Wandb project name.')
flags.DEFINE_integer(
    'log_episodes_every', 10,
    'Log full episode transcripts (game state + LLM responses) every this '
    'many episodes. Set to 0 to disable.')


# ============================================================================
# Game Configurations (mirrors train.py)
# ============================================================================


@dataclasses.dataclass(frozen=True)
class GameConfig:
  """Configuration for an OpenSpiel game."""
  game_name: str
  game_params: dict[str, object]
  num_players: int


_GAME_CONFIGS = {
    'negotiation': GameConfig('negotiation', {}, 2),
    'hanabi': GameConfig('hanabi', {'players': 2}, 2),
    'tiny_hanabi': GameConfig('tiny_hanabi', {}, 2),
}


# ============================================================================
# Lazy HuggingFace imports
# ============================================================================


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


# ============================================================================
# Gemma LLM Backend
# ============================================================================


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
        model_name, token=self._hf_token)
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token

    # ── LoRA adapter ──
    lora_config = peft.LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=['q_proj', 'v_proj'],
        bias='none',
        task_type=peft.TaskType.CAUSAL_LM,
    )

    if use_4bit:
      self.model = peft.prepare_model_for_kbit_training(self.model)

    self.model = peft.get_peft_model(self.model, lora_config)
    self.model.print_trainable_parameters()

    logging.info('Gemma backend ready on device=%s', self.device)

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
    new_tokens = output_ids[0, inputs['input_ids'].shape[1]:]
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
    response_logits = logits[0, prompt_len - 1:-1, :]  # (response_len, vocab)
    response_ids = inputs['input_ids'][0, prompt_len:]  # (response_len,)

    log_probs = torch.log_softmax(response_logits, dim=-1)
    token_log_probs = log_probs.gather(
        1, response_ids.unsqueeze(1)
    ).squeeze(1)  # (response_len,)

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

    response_logits = logits[0, prompt_len - 1:-1, :]
    response_ids = inputs['input_ids'][0, prompt_len:]

    log_probs = torch.log_softmax(response_logits, dim=-1)
    token_log_probs = log_probs.gather(
        1, response_ids.unsqueeze(1)
    ).squeeze(1)

    return token_log_probs.sum()


# ============================================================================
# Trajectory (same structure as train.py but includes prompt text)
# ============================================================================


@dataclasses.dataclass
class RLTrajectoryStep:
  """One decision step in a trajectory, storing data needed for RL updates.

  Attributes:
    prompt: The full text prompt sent to the LLM.
    action_text: The text of the selected action.
    action_id: Integer action ID in the OpenSpiel game.
    log_prob: Log-probability of the action under the policy (float, no grad).
    state_text: The rendered game state (before prompt construction).
    llm_response: The raw text response from the LLM.
    game_action_text: The game's canonical action string.
  """
  prompt: str
  action_text: str
  action_id: int
  log_prob: float
  state_text: str = ''
  llm_response: str = ''
  game_action_text: str = ''


@dataclasses.dataclass
class PlayerTrajectory:
  """Full trajectory for one player in one episode.

  Attributes:
    player_id: The player index.
    steps: List of RLTrajectoryStep objects.
    reward: The final reward for this player.
  """
  player_id: int
  steps: list[RLTrajectoryStep] = dataclasses.field(default_factory=list)
  reward: float = 0.0


# ============================================================================
# Gemma RL Trainer
# ============================================================================


class GemmaRLTrainer:
  """REINFORCE trainer that fine-tunes Gemma 2B via LoRA on OpenSpiel games.

  This trainer:
    1. Runs game episodes using LLMAgents backed by the local Gemma model.
    2. Collects (prompt, action_text) pairs along with rewards.
    3. Computes REINFORCE loss: L = -sum_t[ log_pi(a_t|s_t) * R ]
    4. Back-propagates through the LoRA adapter and updates weights.

  Typical usage:
    ```
    trainer = GemmaRLTrainer(game_name='tiny_hanabi', ...)
    trainer.train()
    ```
  """

  def __init__(
      self,
      game_name: str,
      gemma_backend: GemmaLLMBackend,
      num_episodes: int = 500,
      eval_every: int = 50,
      lr: float = 1e-4,
      max_grad_norm: float = 1.0,
      output_dir: str = '/tmp/teamgamesrl',
  ):
    """Initializes the GemmaRLTrainer.

    Args:
      game_name: Key into _GAME_CONFIGS.
      gemma_backend: The GemmaLLMBackend instance.
      num_episodes: Total training episodes.
      eval_every: Evaluation frequency (in episodes).
      lr: Learning rate for the LoRA adapter optimizer.
      max_grad_norm: Gradient clipping norm.
      output_dir: Directory for logs and checkpoints.

    Raises:
      ValueError: If game_name is not recognized.
    """
    if game_name not in _GAME_CONFIGS:
      raise ValueError(
          f'Unknown game: {game_name}. '
          f'Available: {list(_GAME_CONFIGS.keys())}')

    self.game_config = _GAME_CONFIGS[game_name]
    self.game_name = game_name
    self.num_episodes = num_episodes
    self.eval_every = eval_every
    self.max_grad_norm = max_grad_norm
    self.output_dir = output_dir
    self.backend = gemma_backend

    # ── OpenSpiel environment ──
    from open_spiel.python import rl_environment
    if self.game_config.game_params:
      self.env = rl_environment.Environment(
          self.game_config.game_name, **self.game_config.game_params)
    else:
      self.env = rl_environment.Environment(self.game_config.game_name)

    # ── Per-player renderers and agents ──
    self.renderers = []
    self.agents = []
    for pid in range(self.game_config.num_players):
      renderer = _create_renderer(self.game_config)
      self.renderers.append(renderer)
      agent = llm_agent.LLMAgent(
          player_id=pid,
          renderer=renderer,
          llm=gemma_backend,
          env=self.env,
          temperature=FLAGS.temperature,
      )
      self.agents.append(agent)

    # ── Optimizer (only LoRA params are trainable) ──
    trainable_params = [
        p for p in gemma_backend.model.parameters() if p.requires_grad
    ]
    self.optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    # ── Metrics ──
    self._episode_rewards: list[float] = []
    self._episode_losses: list[float] = []
    self._player_wins = np.zeros(self.game_config.num_players, dtype=np.int64)
    self._total_episodes = 0

    os.makedirs(output_dir, exist_ok=True)
    logging.info(
        'GemmaRLTrainer ready: game=%s, lr=%g, lora_params=%d',
        game_name, lr, sum(p.numel() for p in trainable_params))

  def run_episode(
      self, is_evaluation: bool = False,
  ) -> list[PlayerTrajectory]:
    """Plays one full episode, collecting trajectory data per player.

    Args:
      is_evaluation: If True, use greedy decoding (temperature → 0).

    Returns:
      List of PlayerTrajectory objects, one per player.
    """
    num_players = self.game_config.num_players
    trajectories = [PlayerTrajectory(player_id=p) for p in range(num_players)]

    time_step = self.env.reset()

    while not time_step.last():
      current_player = time_step.current_player()

      # Render state text.
      state = self.env._state  # pylint: disable=protected-access
      state_text = self.renderers[current_player].render_state(
          state, current_player, self.env.game)

      # Get legal actions with descriptions.
      legal_actions_with_desc = self.renderers[current_player] \
          .render_legal_actions(state, current_player, self.env.game)
      legal_actions = [a for a, _ in legal_actions_with_desc]
      action_descriptions = [d for _, d in legal_actions_with_desc]

      # Build prompt.
      prompt = self.agents[current_player]._build_prompt(
          state_text, legal_actions, action_descriptions)

      # Generate action.
      temp = 0.01 if is_evaluation else FLAGS.temperature
      response, log_prob = self.backend.generate_with_logprobs(
          prompt, temperature=temp, max_tokens=64)

      # Parse action.
      action_id = self.renderers[current_player].parse_action(
          response, legal_actions_with_desc)
      if action_id is None:
        action_id = int(np.random.choice(legal_actions))

      action_text = state.action_to_string(current_player, action_id)

      trajectories[current_player].steps.append(RLTrajectoryStep(
          prompt=prompt,
          action_text=response.strip(),
          action_id=action_id,
          log_prob=log_prob,
          state_text=state_text,
          llm_response=response,
          game_action_text=action_text,
      ))

      time_step = self.env.step([action_id])

    # Assign rewards.
    if time_step.rewards is not None:
      for p in range(num_players):
        trajectories[p].reward = time_step.rewards[p]

    return trajectories

  def compute_and_apply_reinforce_loss(
      self, trajectories: list[PlayerTrajectory],
  ) -> float:
    """Computes REINFORCE loss and updates LoRA weights.

    For each step in each player's trajectory:
      loss += -log_pi(action | state) * reward

    The log-probabilities are recomputed with gradients via
    compute_action_log_prob so that back-propagation flows through
    the LoRA adapter.

    Args:
      trajectories: Per-player trajectories from one episode.

    Returns:
      The scalar loss value (float).
    """
    self.optimizer.zero_grad()
    total_loss = torch.tensor(0.0, device=self.backend.device)

    for traj in trajectories:
      if not traj.steps or traj.reward == 0.0:
        continue

      for step in traj.steps:
        # Recompute log-prob *with gradients*.
        log_prob = self.backend.compute_action_log_prob(
            step.prompt, step.action_text)
        total_loss = total_loss + (-log_prob * traj.reward)

    if total_loss.requires_grad:
      total_loss.backward()
      torch.nn.utils.clip_grad_norm_(
          self.backend.model.parameters(), self.max_grad_norm)
      self.optimizer.step()

    return float(total_loss.item())

  def evaluate(self, num_episodes: int = 10) -> dict[str, float]:
    """Evaluates the current policy over multiple episodes.

    Args:
      num_episodes: Number of evaluation episodes.

    Returns:
      Dictionary of evaluation metrics.
    """
    self.backend.model.eval()
    num_players = self.game_config.num_players
    all_rewards = [[] for _ in range(num_players)]
    wins = np.zeros(num_players, dtype=np.int64)

    for _ in range(num_episodes):
      trajectories = self.run_episode(is_evaluation=True)
      rewards = [t.reward for t in trajectories]
      for p in range(num_players):
        all_rewards[p].append(rewards[p])
      max_r = max(rewards)
      winners = [p for p in range(num_players) if rewards[p] == max_r]
      if len(winners) == 1:
        wins[winners[0]] += 1

    self.backend.model.train()

    metrics = {}
    for p in range(num_players):
      pr = np.array(all_rewards[p])
      metrics[f'eval/mean_reward_p{p}'] = float(np.mean(pr))
      metrics[f'eval/win_rate_p{p}'] = float(wins[p] / num_episodes)
    return metrics

  def save_checkpoint(self, episode: int) -> str:
    """Saves a LoRA adapter checkpoint.

    Args:
      episode: Current episode number (used in the checkpoint path).

    Returns:
      Path to the saved checkpoint directory.
    """
    ckpt_dir = os.path.join(self.output_dir, f'checkpoint-ep{episode}')
    self.backend.model.save_pretrained(ckpt_dir)
    self.backend.tokenizer.save_pretrained(ckpt_dir)
    logging.info('Checkpoint saved: %s', ckpt_dir)
    return ckpt_dir

  def _log_episode(
      self,
      episode: int,
      trajectories: list[PlayerTrajectory],
      loss: float,
      is_evaluation: bool = False,
  ) -> None:
    """Logs a full episode transcript to JSONL for visualization.

    Each line in the JSONL file is one episode with per-step details:
    game state, LLM prompt/response, parsed action, reward, etc.

    Args:
      episode: The episode number.
      trajectories: Per-player trajectories from the episode.
      loss: The REINFORCE loss for this episode.
      is_evaluation: Whether this was an evaluation episode.
    """
    log_path = os.path.join(self.output_dir, 'episode_log.jsonl')
    record = {
        'episode': episode,
        'game': self.game_name,
        'is_evaluation': is_evaluation,
        'loss': loss,
        'players': [],
    }
    for traj in trajectories:
      player_data = {
          'player_id': traj.player_id,
          'reward': traj.reward,
          'steps': [],
      }
      for step in traj.steps:
        player_data['steps'].append({
            'state_text': step.state_text,
            'llm_response': step.llm_response,
            'game_action': step.game_action_text,
            'action_id': step.action_id,
            'log_prob': step.log_prob,
        })
      record['players'].append(player_data)

    with open(log_path, 'a') as f:
      f.write(json.dumps(record) + '\n')

  def train(self) -> None:
    """Runs the main REINFORCE training loop.

    For each episode:
      1. Play an episode and collect trajectories.
      2. Compute REINFORCE loss and update LoRA weights.
      3. Log metrics.
      4. Periodically evaluate and checkpoint.
    """
    logging.info(
        'Starting Gemma RL training: %d episodes on %s',
        self.num_episodes, self.game_name)
    start_time = time.time()
    log_every = FLAGS.log_every
    checkpoint_every = FLAGS.checkpoint_every

    # Optional W&B init.
    if FLAGS.use_wandb:
      import wandb  # pylint: disable=g-import-not-at-top
      wandb.init(
          project=FLAGS.wandb_project,
          config={
              'game': self.game_name,
              'model': FLAGS.model_name,
              'lora_rank': FLAGS.lora_rank,
              'lr': FLAGS.lr,
              'temperature': FLAGS.temperature,
              'num_episodes': self.num_episodes,
          },
      )

    self.backend.model.train()

    for ep in range(1, self.num_episodes + 1):
      # ── Episode ──
      trajectories = self.run_episode(is_evaluation=False)
      loss = self.compute_and_apply_reinforce_loss(trajectories)

      # ── Episode logging ──
      log_episodes_every = FLAGS.log_episodes_every
      if log_episodes_every > 0 and ep % log_episodes_every == 0:
        self._log_episode(ep, trajectories, loss)

      # ── Metrics ──
      ep_rewards = [t.reward for t in trajectories]
      mean_reward = float(np.mean(ep_rewards))
      self._episode_rewards.append(mean_reward)
      self._episode_losses.append(loss)
      self._total_episodes += 1

      max_r = max(ep_rewards)
      winners = [
          p for p in range(self.game_config.num_players)
          if ep_rewards[p] == max_r
      ]
      if len(winners) == 1:
        self._player_wins[winners[0]] += 1

      # ── Logging ──
      if ep % log_every == 0:
        elapsed = time.time() - start_time
        avg_r = float(np.mean(self._episode_rewards[-log_every:]))
        avg_l = float(np.mean(self._episode_losses[-log_every:]))
        logging.info(
            'Ep %d/%d | reward=%.4f (avg=%.4f) | loss=%.4f (avg=%.4f) | '
            '%.1f sec elapsed',
            ep, self.num_episodes, mean_reward, avg_r, loss, avg_l, elapsed)

        if FLAGS.use_wandb:
          import wandb  # pylint: disable=g-import-not-at-top
          wandb.log({
              'episode': ep,
              'reward': mean_reward,
              'avg_reward': avg_r,
              'loss': loss,
              'avg_loss': avg_l,
          })

      # ── Evaluation ──
      if ep % self.eval_every == 0:
        logging.info('--- Evaluation at episode %d ---', ep)
        eval_metrics = self.evaluate(
            num_episodes=FLAGS.num_eval_episodes)
        for k, v in sorted(eval_metrics.items()):
          logging.info('  %s: %.4f', k, v)
        if FLAGS.use_wandb:
          import wandb  # pylint: disable=g-import-not-at-top
          wandb.log(eval_metrics, step=ep)

      # ── Checkpoint ──
      if ep % checkpoint_every == 0:
        self.save_checkpoint(ep)

    # ── Final summary ──
    total_time = time.time() - start_time
    logging.info('Training complete: %d episodes in %.1f seconds.',
                 self.num_episodes, total_time)
    logging.info('Final mean reward: %.4f',
                 float(np.mean(self._episode_rewards)))
    logging.info('Final mean loss: %.4f',
                 float(np.mean(self._episode_losses)))
    for p in range(self.game_config.num_players):
      logging.info('  Player %d win rate: %.1f%% (%d/%d)',
                   p,
                   100.0 * self._player_wins[p] / self._total_episodes,
                   self._player_wins[p], self._total_episodes)

    # Save final checkpoint.
    self.save_checkpoint(self.num_episodes)

    if FLAGS.use_wandb:
      import wandb  # pylint: disable=g-import-not-at-top
      wandb.finish()


# ============================================================================
# Helpers
# ============================================================================


def _create_renderer(game_config: GameConfig) -> state_renderers.BaseStateRenderer:
  """Creates a state renderer for the given game."""
  if game_config.game_name in ('tiny_hanabi', 'hanabi'):
    return state_renderers.HanabiRenderer()
  elif game_config.game_name == 'negotiation':
    return state_renderers.NegotiationRenderer()
  else:
    return state_renderers.GenericRenderer()


# ============================================================================
# Entry point
# ============================================================================


def main(argv: list[str]) -> None:
  """Main entry point for Gemma RL training."""
  del argv

  np.random.seed(FLAGS.seed)
  torch.manual_seed(FLAGS.seed)

  logging.info('=== TeamGamesRL — Gemma 2B RL Training ===')
  logging.info('Game: %s', FLAGS.game)
  logging.info('Model: %s (4-bit=%s)', FLAGS.model_name, FLAGS.use_4bit)
  logging.info('LoRA: rank=%d, alpha=%d, dropout=%.2f',
               FLAGS.lora_rank, FLAGS.lora_alpha, FLAGS.lora_dropout)
  logging.info('Training: episodes=%d, lr=%g, temp=%.2f',
               FLAGS.num_episodes, FLAGS.lr, FLAGS.temperature)

  # ── Load model ──
  backend = GemmaLLMBackend(
      model_name=FLAGS.model_name,
      lora_rank=FLAGS.lora_rank,
      lora_alpha=FLAGS.lora_alpha,
      lora_dropout=FLAGS.lora_dropout,
      use_4bit=FLAGS.use_4bit,
      max_seq_len=FLAGS.max_seq_len,
  )

  # ── Train ──
  trainer = GemmaRLTrainer(
      game_name=FLAGS.game,
      gemma_backend=backend,
      num_episodes=FLAGS.num_episodes,
      eval_every=FLAGS.eval_every,
      lr=FLAGS.lr,
      max_grad_norm=FLAGS.max_grad_norm,
      output_dir=FLAGS.output_dir,
  )
  trainer.train()


if __name__ == '__main__':
  app.run(main)

