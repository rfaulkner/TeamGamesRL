# TeamGamesRL

**A modular framework for training LLM agents on cooperative and competitive
multi-agent games using reinforcement learning and
[OpenSpiel](https://github.com/google-deepmind/open_spiel).**

---

## Project Goals

TeamGamesRL explores a novel research direction: using reinforcement learning to
fine-tune large language models so they become better strategic players in
multi-agent games. The framework is designed to be **model-agnostic**,
**algorithm-agnostic**, and **environment-extensible** — swap in any LLM
backend, any RL algorithm, or any OpenSpiel game.

Key research questions:

1. **Can LLMs learn game-theoretic reasoning through RL?** We put LLM agents
   into cooperative and competitive OpenSpiel games and train them with policy
   gradients — does the model learn to propose better deals, give better hints,
   and coordinate more effectively?

2. **How does natural-language action selection compare to discrete policies?**
   Instead of a traditional action-head MLP, our agents read text-rendered game
   states and select actions via text generation. The policy *is* the language
   model itself.

3. **Efficient fine-tuning at scale.** With LoRA adapters and quantization, a
   single GPU can train a multi-billion-parameter model in the RL loop with
   minimal VRAM overhead.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                     TeamGamesRL Pipeline                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌───────────────┐   ┌──────────────────┐   ┌──────────────────┐  │
│  │   OpenSpiel    │   │  State Renderer   │   │   LLM Backend    │  │
│  │  Environment   │──▶│  (text bridge)    │──▶│  (any model)     │  │
│  │  (env/)        │   │  (env/)           │   │  (backend/)      │  │
│  └───────┬───────┘   └──────────────────┘   └───────┬──────────┘  │
│          │                                           │             │
│          │◀─────────── action ID ◀──── parse ◀───────┘             │
│          │                                                         │
│          ▼                                                         │
│  ┌───────────────┐                                                 │
│  │   Trajectory   │──▶ RL Algorithm ──▶ weight update              │
│  │   Collector    │    (learn/)                                     │
│  │ (trainer/)     │                                                │
│  └───────────────┘                                                 │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
TeamGamesRL/
├── backend/                        # Swappable LLM backends
│   ├── __init__.py
│   └── gemma_backend.py            # Gemma 2B with LoRA + 4-bit quantization
│
├── env/                            # Game environments and state rendering
│   ├── __init__.py
│   ├── game_config.py              # GameConfig registry (add new games here)
│   ├── game_env.py                 # Environment + renderer factory functions
│   └── state_renderers.py          # Text ↔ action bridges per game
│
├── learn/                          # RL algorithm implementations
│   ├── __init__.py
│   ├── trajectory.py               # Trajectory data classes
│   ├── reinforce.py                # REINFORCE + baseline + KL penalty
│   └── grpo.py                     # GRPO via TRL's GRPOTrainer
│
├── trainer/                        # Training orchestration + entry points
│   ├── __init__.py
│   ├── rl_trainer.py               # Model-agnostic training loop
│   └── gemma_rl_trainer.py         # Gemma-specific CLI entry point
│
├── llm_agent.py                    # LLMInterface ABC, MockLLM, GeminiLLM, LLMAgent
├── train.py                        # Lightweight trainer for mock/API LLM backends
├── view_episodes.py                # CLI tool to inspect episode logs
├── run_hanabi.sh                    # SLURM submission script (Tiny Hanabi)
├── setup.sh                        # One-command environment bootstrap
└── requirements.txt                # Python dependencies
```

---

## Extensibility

TeamGamesRL is built around four extension points: **backends**, **environments**,
**renderers**, and **algorithms**. Each can be swapped or extended independently.

### Adding a New Model Backend

Create a new file in `backend/` that implements the `LLMInterface` ABC from
`llm_agent.py`:

```python
# backend/my_model_backend.py

import llm_agent

class MyModelBackend(llm_agent.LLMInterface):
    """A custom LLM backend."""

    def __init__(self, model_name: str, **kwargs):
        # Load your model, tokenizer, adapter, etc.
        self.model = ...
        self.tokenizer = ...
        self.device = ...

    def generate(self, prompt: str, temperature: float = 0.8,
                 max_tokens: int = 64) -> str:
        """Generate a text completion."""
        ...

    def generate_with_logprobs(self, prompt: str, temperature: float = 0.8,
                               max_tokens: int = 64) -> tuple[str, float]:
        """Generate text and return (response, log_probability)."""
        ...

    def compute_action_log_prob(self, prompt: str,
                                action_text: str) -> 'torch.Tensor':
        """Recompute log-prob with gradient tracking (for RL training)."""
        ...
```

Then create a corresponding entry point in `trainer/` (or modify an existing
one) to instantiate your backend and pass it to `RLTrainer`:

```python
from backend.my_model_backend import MyModelBackend
from trainer.rl_trainer import RLTrainer

backend = MyModelBackend(model_name='my-org/my-model')
trainer = RLTrainer(game_name='tiny_hanabi', backend=backend, lr=1e-4)
trainer.train_reinforce(config)
```

### Adding a New Game Environment

1. **Register the game** in `env/game_config.py`:

```python
MY_GAME_CONFIG = GameConfig(
    game_name='my_openspiel_game',  # Must match the OpenSpiel game name
    game_params={'players': 3},     # Game-specific parameters
    num_players=3,
)

_GAME_CONFIGS['my_game'] = MY_GAME_CONFIG
```

2. **Create a renderer** in `env/state_renderers.py` by subclassing
   `BaseStateRenderer`:

```python
class MyGameRenderer(BaseStateRenderer):

    def render_state(self, state, player_id, game) -> str:
        """Convert the game state to a natural-language description."""
        ...

    def render_legal_actions(self, state, player_id, game):
        """Return list of (action_id, description) for legal actions."""
        ...

    def parse_action(self, llm_response, legal_actions_with_desc):
        """Parse the LLM's text response into an action ID."""
        ...
```

3. **Register the renderer** in the `get_renderer()` factory at the bottom of
   `env/state_renderers.py`.

### Adding a New RL Algorithm

Create a new module in `learn/`. The algorithm should follow one of two
patterns:

**Pattern A — Updater** (for online, per-episode algorithms like REINFORCE):

```python
# learn/my_algorithm.py

@dataclasses.dataclass
class MyAlgorithmConfig:
    lr: float = 1e-4
    ...

class MyAlgorithmUpdater:
    def __init__(self, backend, optimizer, config):
        ...

    def update(self, trajectories: list[PlayerTrajectory]) -> float:
        """Compute loss and apply one gradient step. Return loss value."""
        ...

    def flush(self):
        """Flush any accumulated state (e.g. gradient accumulation)."""
        ...
```

Then add a `train_my_algorithm()` method to `trainer/rl_trainer.py` following
the pattern of `train_reinforce()`.

**Pattern B — Runner** (for batch algorithms like GRPO that manage their own
training loop):

```python
# learn/my_batch_algorithm.py

class MyBatchRunner:
    def __init__(self, env, renderers, agents, backend, game_config,
                 evaluate_fn, save_checkpoint_fn, output_dir, config):
        ...

    def run(self):
        """Execute the full training procedure."""
        ...
```

Then add a `train_my_algorithm()` method to `trainer/rl_trainer.py` that
instantiates and calls `runner.run()`, following the pattern of `train_grpo()`.

---

## Supported Games

| Game | Players | Type | Description |
|---|---|---|---|
| `tiny_hanabi` | 2 | Cooperative | A minimal Hanabi — great for fast iteration and debugging. |
| `hanabi` | 2 | Cooperative | Full Hanabi — imperfect information, hints, and fireworks. |
| `negotiation` | 2 | Competitive | Multi-item deal-making — propose splits, send utterances, accept/reject. |

Any [OpenSpiel game](https://github.com/google-deepmind/open_spiel/blob/master/docs/games.md)
can be added by registering a `GameConfig` and writing a `StateRenderer`.

---

## Getting Started

### 1. Clone and set up the environment

```bash
cd TeamGamesRL
source setup.sh
```

This creates a `.venv` virtualenv, installs all dependencies, and optionally
logs you into Hugging Face (required for gated models like Gemma).

### 2. Quick test with the mock LLM (no GPU needed)

```bash
python train.py --game=tiny_hanabi --llm_type=mock --num_episodes=100
```

This validates the full pipeline using a random-action agent.

### 3. Train with Gemma 2B + LoRA (requires GPU)

```bash
# REINFORCE
python trainer/gemma_rl_trainer.py \
  --rl_algorithm=reinforce \
  --game=tiny_hanabi \
  --num_episodes=500 \
  --lr=1e-4

# GRPO (TRL)
python trainer/gemma_rl_trainer.py \
  --rl_algorithm=grpo \
  --game=tiny_hanabi \
  --grpo_passes=10 \
  --grpo_collect_episodes=50

# Full configuration
python trainer/gemma_rl_trainer.py \
  --game=hanabi \
  --model_name=google/gemma-2-2b \
  --lora_rank=32 \
  --lora_alpha=64 \
  --lr=5e-5 \
  --temperature=0.8 \
  --num_episodes=2000 \
  --eval_every=100 \
  --checkpoint_every=200 \
  --use_wandb \
  --output_dir=/tmp/teamgamesrl/hanabi_run1
```

### 4. Submit via SLURM

```bash
# GRPO (default: tiny_hanabi, 30 passes)
sbatch scripts/run_grpo.sh tiny_hanabi google/gemma-2-2b 16 3e-5 30 50

# REINFORCE (default: tiny_hanabi, 500 episodes)
sbatch scripts/run_reinforce.sh tiny_hanabi google/gemma-2-2b 16 1e-4 500
```

### 5. Resume from a checkpoint

LoRA checkpoints are saved to `--output_dir` every `--checkpoint_every`
episodes. To resume, load the adapter from the checkpoint directory
(HuggingFace PEFT standard format).

---

## Interpreting Results

Training produces several output files in `--output_dir`:

### Training Metrics (`results/training_metrics.csv`)

Logged every `--log_every` episodes:

| Column | Description |
|---|---|
| `episode` | Episode number |
| `reward` | Mean reward across all players for this episode |
| `loss` | RL loss for this episode |
| `avg_reward` | Rolling average reward over the last `log_every` episodes |
| `avg_loss` | Rolling average loss |
| `elapsed_sec` | Wall-clock time since training started |

**What to look for:**
- **`avg_reward` trending upward** indicates the agents are learning to play
  better. For cooperative games like Hanabi, all players share the reward, so
  this reflects team performance.
- **`avg_loss` decreasing then stabilizing** is normal. Very large or erratic
  loss values may indicate the learning rate is too high.
- **Reward plateaus** may indicate the agents have converged, or that
  exploration (temperature) needs adjustment.

### Evaluation Metrics (`results/eval_metrics.csv`)

Logged every `--eval_every` episodes using greedy decoding (near-zero
temperature):

| Column | Description |
|---|---|
| `eval/mean_reward_pN` | Mean reward for player N across eval episodes |
| `eval/win_rate_pN` | Win rate for player N |

**What to look for:**
- **Eval reward > training reward** is expected since eval uses greedy decoding
  (less exploration noise).
- **Balanced `win_rate` across players** in competitive games means neither
  player dominates. Imbalance may indicate one player's policy is
  over-optimized.
- **Eval reward diverging from training reward** can indicate overfitting to
  the training exploration pattern.

### Episode Logs (`episode_log.jsonl`)

Detailed per-step transcripts logged every `--log_episodes_every` episodes.
Each line is a JSON object containing the full game state, LLM prompts,
responses, parsed actions, and rewards for every player. Use the viewer:

```bash
python view_episodes.py --log_file=/tmp/teamgamesrl/episode_log.jsonl
python view_episodes.py --log_file=/tmp/teamgamesrl/episode_log.jsonl --episode=42
```

**What to look for:**
- **Action quality** — Are the agents choosing sensible actions given the game
  state? Early training should show mostly random-seeming choices; later
  episodes should show strategic behavior.
- **Prompt understanding** — Is the LLM parsing the state correctly and
  responding with valid action text?
- **Coordination** (cooperative games) — Are players' actions becoming more
  complementary over time?

### Final Summary (`results/summary.json`)

A JSON snapshot written at the end of training with aggregate statistics:

```json
{
  "game": "tiny_hanabi",
  "num_episodes": 500,
  "total_time_sec": 1234.5,
  "final_mean_reward": 7.82,
  "last_10_mean_reward": 8.45,
  "player_win_rates": {"player_0": 45.2, "player_1": 42.8},
  "team_win_rate": 72.4
}
```

### Weights & Biases (optional)

Pass `--use_wandb` to stream all training and evaluation metrics to W&B in
real time. Useful for comparing runs across hyperparameters, games, or models.

---

## Key Flags

| Flag | Default | Description |
|---|---|---|
| `--rl_algorithm` | `grpo` | `reinforce` or `grpo` |
| `--game` | `tiny_hanabi` | OpenSpiel game to train on |
| `--model_name` | `google/gemma-2-2b` | HuggingFace model ID |
| `--num_episodes` | `500` | Total training episodes |
| `--lr` | `1e-5` | Learning rate |
| `--lora_rank` | `16` | LoRA decomposition rank |
| `--lora_alpha` | `32` | LoRA scaling factor |
| `--use_4bit` | `True` | 4-bit NF4 quantization |
| `--temperature` | `0.8` | Sampling temperature |
| `--eval_every` | `50` | Evaluation frequency |
| `--checkpoint_every` | `100` | Checkpoint frequency |
| `--kl_coeff` | `0.05` | KL penalty against reference model |
| `--gradient_accumulation_steps` | `8` | Episodes to accumulate before update |
| `--use_wandb` | `False` | Enable W&B experiment tracking |

**GRPO-specific:**

| Flag | Default | Description |
|---|---|---|
| `--grpo_passes` | `10` | Number of collect → train rounds |
| `--grpo_collect_episodes` | `50` | Episodes per prompt collection round |
| `--grpo_num_generations` | `4` | Completions per prompt (group size K) |
| `--grpo_max_completion_length` | `64` | Max tokens per GRPO completion |

---

## Dependence on OpenSpiel

TeamGamesRL is built on [OpenSpiel](https://github.com/google-deepmind/open_spiel)
(≥ 1.5), Google DeepMind's framework for research in games. We depend on it for:

- **Game definitions** — cooperative and competitive multi-player games with
  well-defined state spaces and action encodings.
- **RL environment wrapper** — `rl_environment.Environment` provides the
  standard `reset()` / `step()` loop.
- **Game introspection** — `pyspiel.Game` and `pyspiel.State` for action
  descriptions and observation strings.

All OpenSpiel imports are from the public `open_spiel` and `pyspiel` packages
available via `pip install open-spiel`.

---

## Requirements

- **Python 3.11+**
- **CUDA 12.2** (for GPU training; CPU fallback is supported)
- **~6 GB VRAM** with 4-bit quantization + LoRA rank 16
- **Hugging Face account** with access to gated models (e.g.
  [google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b))

---

## License

Apache License 2.0 — see individual source files for details.
