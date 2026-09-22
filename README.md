# TeamGamesRL

**A modular framework for training LLM agents on cooperative and competitive
multi-agent games using reinforcement learning, behavioral cloning, and
[OpenSpiel](https://github.com/google-deepmind/open_spiel) /
[Hanabi Learning Environment (HLE)](https://github.com/google-deepmind/hanabi-learning-environment).**

---

## Project Goals

TeamGamesRL explores a novel research direction: using reinforcement learning to
fine-tune large language models so they become strategic, communicative players in
multi-agent games. The framework is designed to be **model-agnostic**,
**algorithm-agnostic**, and **environment-extensible** — swap in any LLM
backend, any RL algorithm, or any multi-agent game.

Key research questions:

1. **Can LLMs learn game-theoretic conventions and reasoning through RL?**
   We place LLM agents into imperfect-information cooperative games like Hanabi
   and train them with policy gradients (REINFORCE, GRPO) — does the model learn
   to give informative hints, coordinate conventions, and prevent disastrous discards?

2. **Natural-language action selection vs. discrete heads.**
   Instead of an auxiliary discrete action head, our agents read natural-language
   game state prompts and produce actions via autoregressive text generation (optionally
   preceded by deliberative `<think>...</think>` Chain-of-Thought reasoning).
   The language model *is* the policy.

3. **Sample-efficient fine-tuning at scale.**
   With LoRA adapters and 4-bit (NF4) quantization, models from Gemma 2B to Gemma 3 12B
   can be trained in the RL loop on a single workstation or cluster GPU (e.g. L40S, A100).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TeamGamesRL Pipeline                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐       ┌──────────────────┐     ┌────────────────────┐  │
│  │   Game Engine   │       │  State Renderer  │     │    LLM Backend     │  │
│  │ (OpenSpiel/HLE) │──────▶│  (text bridge)   │────▶│ (Gemma 2B/12B LoRA)│  │
│  │ (env/, env/hanabi)      │  (env/, env/hanabi)    │ (backend/gemma_...)│  │
│  └────────┬────────┘       └──────────────────┘     └─────────┬──────────┘  │
│           │                                                   │             │
│           │◀────────────── action ID ◀──── parse ◀────────────┘             │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │   Trajectory /  │──────▶ RL Algorithm ──────▶ LoRA parameter update     │
│  │   GRPO Groups   │        (learn/grpo_sampled)                            │
│  │ (trainer/, learn)                                                        │
│  └─────────────────┘                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
TeamGamesRL/
├── backend/                        # Swappable LLM backends
│   ├── __init__.py
│   └── gemma_backend.py            # Gemma 2B/12B with LoRA + 4-bit quantization
│
├── env/                            # Game environments and state rendering
│   ├── __init__.py
│   ├── game_config.py              # GameConfig registry (add new games here)
│   ├── game_env.py                 # Environment + renderer factory functions
│   ├── state_renderers.py          # Text ↔ action bridges per game
│   └── hanabi/                     # Dedicated Hanabi environment & bots
│       ├── __init__.py
│       ├── hanabi_env.py           # Fast HLE adapter, state serialization & regexes
│       ├── heuristic_player.py     # SafePlayPlayer & SafeBeliefLookaheadPlayer bots
│       ├── belief_expert.py        # Bayesian belief tracker over hidden cards
│       ├── determinize.py          # Determinization / imperfect-info rollouts
│       └── eval_metrics.py         # Multi-game evaluation metrics collector
│
├── data/                           # Demonstration datasets and generators
│   └── generate_bc_data.py         # Generates expert games for BC warm-start
│
├── learn/                          # RL algorithm implementations
│   ├── __init__.py
│   ├── trajectory.py               # Trajectory and step data classes
│   ├── reinforce.py                # REINFORCE + baseline + KL penalty
│   ├── grpo.py                     # Minimal TRL GRPO wrapper
│   ├── grpo_config.py              # Configuration dataclass for sampled GRPO
│   ├── grpo_sampled.py             # Main multi-turn sampled GRPO runner
│   ├── action_reward.py            # Immediate & dense rollout reward shaping
│   └── strategic_actions.py        # Strategic action generation & diversity injection
│
├── trainer/                        # Training orchestration + entry points
│   ├── __init__.py
│   ├── rl_trainer.py               # Model-agnostic training loop & batched eval
│   └── gemma_rl_trainer.py         # Primary CLI entry point for Gemma models
│
├── scripts/                        # Cluster runner scripts
│   ├── run_bc.sh                   # SLURM script for Behavioral Cloning (BC)
│   ├── run_grpo.sh                 # SLURM script for standard GRPO
│   └── run_reinforce.sh            # SLURM script for REINFORCE
│
├── run_hanabi_full.sh              # Full 5-color Hanabi RL SLURM submission script
├── train_bc.py                     # Supervised Fine-Tuning (SFT / BC) trainer
├── train.py                        # Lightweight trainer for mock/API LLM backends
├── llm_agent.py                    # LLMInterface ABC, MockLLM, GeminiLLM, LLMAgent
├── view_episodes.py                # CLI tool to inspect episode transcripts
├── setup.sh                        # One-command environment bootstrap
└── requirements.txt                # Python dependencies
```

---

## Supported Games

| Game | Engine | Players | Type | Description |
|---|---|---|---|---|
| `tiny_hanabi` | OpenSpiel | 2 | Cooperative | Minimal Hanabi (2 colors, 2 cards/hand) — great for fast debugging. |
| `hanabi` | HLE / OpenSpiel | 2 | Cooperative | Full Hanabi (5 colors, 5 ranks, 50-card deck, 8 info tokens, 3 lives). |
| `negotiation` | OpenSpiel | 2 | Competitive | Multi-item deal-making — propose splits, send utterances, accept/reject. |

Any [OpenSpiel game](https://github.com/google-deepmind/open_spiel/blob/master/docs/games.md)
can be added by registering a `GameConfig` in `env/game_config.py` and writing a
`StateRenderer` in `env/state_renderers.py`.

---

## Training Workflows

TeamGamesRL supports two primary training workflows:

### 1. Two-Stage Curriculum: Behavioral Cloning (BC) Warm-Start → GRPO RL

For complex games with sparse or delayed rewards like full Hanabi, cold-starting RL
can spend many passes exploring invalid or uncoordinated actions. We provide a two-stage
curriculum:

1. **Step 1 — Behavioral Cloning (BC)**: Collect demonstrations from an expert
   belief-search bot (`SafeBeliefLookaheadPlayer`), then train a LoRA adapter via SFT:
   ```bash
   # Generate BC demonstrations and fine-tune Gemma 2B or Gemma 3 12B
   sbatch scripts/run_bc.sh --model=google/gemma-3-12b-it --epochs=3 --reasoning
   ```
2. **Step 2 — GRPO Fine-Tuning**: Initialize RL with the pre-trained BC adapter:
   ```bash
   sbatch run_hanabi_full.sh \
     --initial_lora_checkpoint=checkpoints/bc_google_gemma-3-12b-it_<JOB_ID>/best_adapter \
     --reasoning \
     --bot_partner \
     --bot_type=belief_lookahead
   ```

### 2. Direct Reinforcement Learning (GRPO or REINFORCE)

Train directly from base model weights using online episode collection:

```bash
# Submit full Hanabi GRPO on a cluster
sbatch run_hanabi_full.sh --profile=full --grpo_passes=50 --collect=20

# Run locally or interactively on a GPU
python3 trainer/gemma_rl_trainer.py \
  --rl_algorithm=grpo \
  --game=hanabi \
  --model_name=google/gemma-3-12b-it \
  --grpo_passes=25 \
  --grpo_collect_episodes=20 \
  --grpo_num_generations=8 \
  --reward_simulation_mode=dense_chain \
  --eval_batch_size=4
```

---

## Key Features & Algorithms

### Group Relative Policy Optimization (GRPO)
- **Prompt Collection**: Collects multi-turn decision prompts from self-play or bot-partner episodes.
- **Group Generation (K completions)**: For each pivot state, samples $K$ candidate actions.
- **Action Diversity & Substitution**: Freely samples completions, parses actions, and replaces duplicate slots with strategic alternatives (`--strategic_action_mode=substitute`) so groups have diverse rewards to differentiate.
- **Batch-Level Advantage Scaling**: Normalizes advantages across the entire training batch (`--grpo_scale_rewards=batch`) rather than within-group, avoiding amplified gradients on near-zero rollout noise.

### Partner Bots & Self-Play
- **Self-Play**: LLM vs. LLM training.
- **Bot Partner Training** (`--bot_partner`): Alternates roles across episodes (P0=LLM / P1=Bot, and P0=Bot / P1=LLM).
- **Available Bots** (`--bot_type`):
  - `belief_lookahead` (`SafeBeliefLookaheadPlayer`): Maintains Bayesian belief distributions over hidden cards, performing 1-step lookahead without asymmetric partner assumptions. Ideal partner for inducing self-play coordination.
  - `safe_play` (`SafePlayPlayer`): Rule-based baseline prioritizing 100% safe plays, urgent saves, and basic discards.

### Deliberative Chain-of-Thought Reasoning (`--reasoning`)
When enabled, the prompt instructs the model to generate a structured 5-point deliberation inside `<think>...</think>` tags before emitting the final action:
1. Fireworks & token status summary.
2. Own hand evaluation (confirmed playable cards vs. unknowns).
3. Partner hand evaluation (urgent saves vs. play clues).
4. Discard evaluation (safest discard given discard pile).
5. Best action selection.

The parser extracts the final action text outside the `<think>` block, while the full token sequence is trained with policy gradients.

### Fast Batched Evaluation Inference (`--eval_batch_size`)
During periodic evaluation (`--eval_every`), multiple games are stepped simultaneously in lockstep. At each turn, prompt generations for all active agents are batched into a single GPU forward pass, delivering a **~4x speedup** on eval rounds.

---

## Getting Started

### 1. Clone and set up the environment

```bash
cd TeamGamesRL
source setup.sh
```

This sets up a `.venv` virtualenv, installs required libraries (`open-spiel`, `hanabi-learning-environment`, `transformers`, `peft`, `bitsandbytes`, `trl`), and configures Hugging Face access.

### 2. Quick test with Mock LLM (CPU-friendly)

```bash
python3 train.py --game=tiny_hanabi --llm_type=mock --num_episodes=100
```

### 3. Run Unit and Smoke Tests

```bash
python3 -m unittest env/hanabi/smoke_test.py
python3 -m unittest env/hanabi/determinize_test.py
```

---

## Configuration & Key Flags

### General Flags

| Flag | Default | Description |
|---|---|---|
| `--game` | `tiny_hanabi` | Game identifier (`tiny_hanabi`, `hanabi`, `negotiation`). |
| `--model_name` | `google/gemma-2-2b` | Hugging Face model identifier (e.g. `google/gemma-3-12b-it`). |
| `--initial_lora_checkpoint` | `None` | Path to pre-trained LoRA adapter (e.g. from BC warm-start). |
| `--lora_rank` | `16` | LoRA rank (typical: 16 or 32). |
| `--lora_alpha` | `32` | LoRA alpha scaling factor (usually $2 \times \text{rank}$). |
| `--use_4bit` | `True` | Enable 4-bit NF4 quantization for base model. |
| `--lr` | `3e-5` | Learning rate for LoRA parameters. |
| `--temperature` | `0.8` | Generation temperature. |
| `--temperature_anneal_end` | `None` | Anneal temperature toward this value over training passes. |
| `--eval_every` | `50` | Evaluation frequency (in passes or episodes). |
| `--num_eval_episodes` | `10` | Number of evaluation episodes per eval round. |
| `--eval_batch_size` | `4` | Parallel game batch size for greedy evaluation (~4x speedup). |
| `--use_wandb` | `False` | Enable Weights & Biases logging. |
| `--output_dir` | `/tmp/teamgamesrl` | Output directory for checkpoints, metrics, and logs. |

### GRPO & Reward Simulation Flags

| Flag | Default | Description |
|---|---|---|
| `--grpo_passes` | `25` | Number of collect → train rounds. |
| `--grpo_collect_episodes` | `50` | Episodes collected per training pass. |
| `--grpo_num_generations` | `8` | Candidate completions sampled per prompt ($K$). |
| `--grpo_max_completion_length`| `16` | Max tokens per generation (increase to 128+ if `--reasoning` is set). |
| `--reward_simulation_mode` | `rollout` | `dense_chain`, `rollout`, `random`, or `dense`. |
| `--reward_blend_weight` | `0.0` | Weight $w \in [0, 1]$ blending game outcome with step reward. |
| `--reward_survival_exponent` | `0.0` | Convex penalty on lost lives: $(\text{lives} / \text{max\_lives})^E$. |
| `--reward_turn_discount` | `1.0` | Multiplicative discount $\gamma^T \times \text{score}$ penalizing stalling. |
| `--reward_policy_turns` | `1` | Turns played by policy before heuristic rollout completes game. |
| `--grpo_scale_rewards` | `batch` | Advantage normalization: `batch`, `group`, or `none`. |
| `--bot_partner` | `False` | Partner with a bot during collection and evaluation. |
| `--bot_type` | `belief_lookahead`| Bot partner: `belief_lookahead` or `safe_play`. |
| `--reasoning` | `False` | Enable 5-stage deliberative Chain-of-Thought `<think>` prompt. |
| `--strategic_action_selection`| `False`| Inject diverse strategic actions into candidate groups. |
| `--strategic_action_mode` | `substitute` | `substitute` duplicate slots or force via `logits`. |

---

## Output Metrics & Artifacts

Training logs several artifacts to `--output_dir`:

- **`results/training_metrics.csv`**: Logs episode, rewards, loss, rolling averages, and elapsed time.
- **`results/eval_metrics.csv`**: Logs greedy evaluation metrics across pairings (LLM+LLM, LLM+Bot, Bot+LLM), including mean score, bomb rate, and information token efficiency.
- **`episode_log.jsonl`**: Step-by-step game transcripts including raw prompt strings, model completions, parsed action IDs, and rewards. Inspect using:
  ```bash
  python3 view_episodes.py --log_file=<OUTPUT_DIR>/episode_log.jsonl --episode=10
  ```
- **Checkpoints**: Saved to `<OUTPUT_DIR>/checkpoint_pass_*` containing standard Hugging Face PEFT LoRA adapter weights.

---

## Requirements

- **Python 3.11+**
- **CUDA 12.2** (for GPU training; CPU fallback is supported)
- **~6 GB VRAM** with 4-bit quantization + LoRA rank 16 (Gemma 2B) or **~24-48 GB VRAM** (Gemma 3 12B)
- **Hugging Face account** with access to gated models (e.g. `google/gemma-3-12b-it`)

---

## License

Apache License 2.0 — see individual source files for details.
