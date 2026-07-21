# TeamGamesRL

**Multi-agent reinforcement learning on cooperative and competitive team games,
powered by LLMs (Gemma 2B) and [OpenSpiel](https://github.com/google-deepmind/open_spiel).**

---

## Project Goals

TeamGamesRL explores a novel research direction: using reinforcement learning to
fine-tune large language models so they become better strategic players in
multi-agent team games.  The key questions we investigate are:

1. **Can LLMs learn game-theoretic reasoning through RL?**  We put Gemma 2B
   agents into cooperative and competitive OpenSpiel games and train them with
   REINFORCE policy gradients — does the model learn to propose better deals,
   give better hints, and coordinate more effectively?

2. **How does natural-language action selection compare to discrete policies?**
   Instead of a traditional action-head MLP, our agents read text-rendered game
   states and select actions via text generation.  The policy is the language
   model itself.

3. **Efficient fine-tuning at scale.**  We use LoRA adapters and 4-bit
   quantization so that a single GPU can train a 2B-parameter model in the RL
   loop with minimal VRAM overhead (~6 GB).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      TeamGamesRL Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────┐    ┌─────────────────┐   ┌───────────────┐  │
│  │   OpenSpiel    │    │  State Renderer  │   │  Gemma 2B +   │  │
│  │  Environment   │───▶│  (text bridge)   │──▶│  LoRA Agent   │  │
│  │  (Hanabi, …)   │    │                  │   │               │  │
│  └───────┬───────┘    └─────────────────┘   └───────┬───────┘  │
│          │                                          │          │
│          │◀────────── action ID ◀───── parse ◀──────┘          │
│          │                                                     │
│          ▼                                                     │
│  ┌───────────────┐                                             │
│  │   Trajectory   │──▶ REINFORCE loss ──▶ LoRA weight update   │
│  │   Collector    │                                            │
│  └───────────────┘                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Modules

| File | Description |
|---|---|
| [`setup.sh`](setup.sh) | One-command environment bootstrap: loads system modules, creates virtualenv, installs all deps, optionally logs into Hugging Face. |
| [`requirements.txt`](requirements.txt) | Python dependencies (OpenSpiel, PyTorch, Transformers, PEFT, TRL, bitsandbytes, etc.). |
| [`gemma_rl_trainer.py`](gemma_rl_trainer.py) | **Main RL training script.** Loads Gemma 2B with LoRA, runs game episodes, computes REINFORCE loss, and updates the adapter. Supports checkpointing, evaluation, and W&B logging. |
| [`train.py`](train.py) | Original REINFORCE training loop using the mock / API-based LLM backends.  Good for rapid prototyping and pipeline validation without GPU. |
| [`llm_agent.py`](llm_agent.py) | Agent classes: `LLMAgent` (prompt-based action selection), `MockLLM` (random baseline), `GeminiLLM` (API backend), `RandomAgent`. |
| [`state_renderers.py`](state_renderers.py) | Text-action bridge — converts OpenSpiel states to natural language prompts and parses LLM output back to action IDs.  Game-specific renderers: `NegotiationRenderer`, `HanabiRenderer`, `GenericRenderer`. |
| [`.gitignore`](.gitignore) | Standard Python + logs ignore patterns. |

---

## Dependence on OpenSpiel

TeamGamesRL is built on top of [OpenSpiel](https://github.com/google-deepmind/open_spiel)
(≥ 1.5), Google DeepMind's framework for research in games.  We depend on it for:

- **Game definitions** — cooperative (Hanabi, Tiny Hanabi) and competitive
  (Negotiation) multi-player games with well-defined state spaces and action
  encodings.
- **RL environment wrapper** — `open_spiel.python.rl_environment.Environment`
  provides the standard `reset()` / `step()` loop with `TimeStep` objects.
- **Agent interface** — `open_spiel.python.rl_agent.StepOutput` structures
  used by `LLMAgent.step()`.
- **Game introspection** — `pyspiel.Game` and `pyspiel.State` for action
  descriptions, observation strings, and game parameters.

All OpenSpiel imports are from the public `open_spiel` and `pyspiel` packages
available via `pip install open-spiel`.

---

## Supported Games

| Game | Players | Type | Description |
|---|---|---|---|
| `tiny_hanabi` | 2 | Cooperative | A minimal version of Hanabi — great for fast iteration and debugging. |
| `hanabi` | 2 | Cooperative | Full Hanabi — a cooperative card game with imperfect information, hints, and fireworks. |
| `negotiation` | 2 | Competitive | Multi-item deal-making — players propose item splits, send utterances, and accept/reject. |

---

## Getting Started

### 1. Clone and set up the environment

```bash
cd TeamGamesRL
source setup.sh
```

This creates a `.venv` virtualenv, installs all dependencies, and optionally
logs you into Hugging Face (required for the gated Gemma model).

### 2. Quick test with the mock LLM (no GPU needed)

```bash
python train.py --game=tiny_hanabi --llm_type=mock --num_episodes=100
```

This validates the full pipeline using a random action agent.

### 3. Train Gemma 2B with LoRA (requires GPU + HF access)

```bash
# Basic run
python gemma_rl_trainer.py \
  --game=tiny_hanabi \
  --num_episodes=500 \
  --lr=1e-4

# Full configuration
python gemma_rl_trainer.py \
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

### 4. Resume from a checkpoint

LoRA checkpoints are saved to `--output_dir` every `--checkpoint_every`
episodes.  To resume, load the adapter from the checkpoint directory
(HuggingFace PEFT standard format).

---

## Key Flags

| Flag | Default | Description |
|---|---|---|
| `--game` | `tiny_hanabi` | OpenSpiel game to train on |
| `--model_name` | `google/gemma-2-2b` | HuggingFace model ID |
| `--num_episodes` | `500` | Total training episodes |
| `--lr` | `1e-4` | LoRA adapter learning rate |
| `--lora_rank` | `16` | LoRA decomposition rank |
| `--lora_alpha` | `32` | LoRA scaling factor |
| `--use_4bit` | `True` | 4-bit NF4 quantization |
| `--temperature` | `0.8` | Sampling temperature |
| `--eval_every` | `50` | Evaluation frequency |
| `--checkpoint_every` | `100` | Checkpoint frequency |
| `--use_wandb` | `False` | Enable W&B experiment tracking |
| `--max_seq_len` | `512` | Max sequence length for tokenization |
| `--max_grad_norm` | `1.0` | Gradient clipping norm |

---

## Requirements

- **Python 3.11+**
- **CUDA 12.2** (for GPU training; CPU fallback is supported)
- **~6 GB VRAM** with 4-bit quantization + LoRA rank 16
- **Hugging Face account** with access to [google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b)

---

## License

Apache License 2.0 — see individual source files for details.
