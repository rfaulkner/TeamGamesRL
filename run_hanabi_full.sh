#!/bin/bash
#SBATCH --job-name=hanabi-full-rl
#SBATCH --account=aip-rgrosse
#SBATCH --output=slurm/output/%j_%x.out
#SBATCH --error=slurm/output/%j_%x.err

#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

# ============================================================================
# Full Hanabi RL — SLURM submission script
#
# Trains Gemma 3 12B with GRPO on full (5-color, 5-rank, 2-player) Hanabi.
# This is the "real" Hanabi setup — 30-40 turn episodes with delayed rewards.
#
# Differences from tiny_hanabi (run_hanabi.sh):
#   - Game:       hanabi (not tiny_hanabi)
#   - Model:      Gemma 3 12B-IT (not Gemma 2 2B)
#   - Seq len:    2048 (not 512)
#   - Completion: 32 tokens (not 16) — full Hanabi actions are longer
#   - Episodes:   fewer per pass (each is ~15x longer)
#   - Time:       4 hours (not 2)
#   - Memory:     48 GB (12B model needs more headroom)
#
# Usage:
#   sbatch run_hanabi_full.sh                         # all defaults
#   sbatch run_hanabi_full.sh --grpo_passes=100       # longer training
#   sbatch run_hanabi_full.sh --lr=1e-5               # override LR
#   sbatch run_hanabi_full.sh --model=google/gemma-2-2b  # test with smaller model
#
# Flags (all optional, order doesn't matter):
#   --model=ID          HF model identifier    (default: google/gemma-3-12b-it)
#   --lora_rank=R       LoRA rank              (default: 16)
#   --lr=RATE           Learning rate          (default: 2e-5)
#   --grpo_passes=P     GRPO passes            (default: 50)
#   --collect=N         Episodes per pass      (default: 20)
#   --max_seq_len=L     Max sequence length    (default: 2048)
#   --profile=MODE      quick|full             (default: full)
# ============================================================================

set -euo pipefail

# ── Parse arguments with defaults ────────────────────────────────────────────

GAME="hanabi"
MODEL_ID="google/gemma-3-12b-it"
LORA_RANK=16
LR="2e-5"
GRPO_PASSES=50
COLLECT_EPISODES=20
MAX_SEQ_LEN=2048
PROFILE="full"
EXTRA_FLAGS=""

for arg in "$@"; do
  case "$arg" in
    --model=*)        MODEL_ID="${arg#*=}" ;;
    --lora_rank=*)    LORA_RANK="${arg#*=}" ;;
    --lr=*)           LR="${arg#*=}" ;;
    --grpo_passes=*)  GRPO_PASSES="${arg#*=}" ;;
    --collect=*)      COLLECT_EPISODES="${arg#*=}" ;;
    --max_seq_len=*)  MAX_SEQ_LEN="${arg#*=}" ;;
    --profile=*)      PROFILE="${arg#*=}" ;;
    --help|-h)
      echo "Usage: sbatch run_hanabi_full.sh [--model=M] [--lora_rank=R] [--lr=L] [--grpo_passes=P] [--collect=N] [--max_seq_len=L] [--profile=quick|full] [--extra...]"
      exit 0 ;;
    --*) EXTRA_FLAGS="${EXTRA_FLAGS} ${arg}" ;;
    *) echo "Unknown flag: $arg (try --help)"; exit 1 ;;
  esac
done

# ── Profile presets ──────────────────────────────────────────────────────────
# Quick mode: ~1 hour iteration runs for testing approach and debugging.
# Full mode:  ~4 hour comprehensive runs for real experiments.

if [ "$PROFILE" = "quick" ]; then
  GRPO_PASSES="${GRPO_PASSES:-15}"
  COLLECT_EPISODES="${COLLECT_EPISODES:-10}"
  NUM_GENERATIONS=4
  NUM_EVAL_EPISODES=5
  echo "[PROFILE] Quick mode: ${GRPO_PASSES} passes, ${COLLECT_EPISODES} eps/pass, K=${NUM_GENERATIONS}"
else
  NUM_GENERATIONS=4
  NUM_EVAL_EPISODES=10
  echo "[PROFILE] Full mode: ${GRPO_PASSES} passes, ${COLLECT_EPISODES} eps/pass, K=${NUM_GENERATIONS}"
fi

# ── Derived settings ─────────────────────────────────────────────────────────

project_dir="/home/$USER/projects/aip-rgrosse/$USER/TeamGamesRL"
output_dir="/scratch/$USER/teamgamesrl/${GAME}_gemma3-12b_lr${LR}_rank${LORA_RANK}_passes${GRPO_PASSES}_${SLURM_JOB_ID}"

export HF_HOME="/scratch/$USER/hf_cache"
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export PYTHONPATH="${project_dir}:${PYTHONPATH:-.}"

# ── Hugging Face auth (Gemma 3 is a gated model) ────────────────────────────
export HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null || true)}"

# ── Load system modules ─────────────────────────────────────────────────────

module load python/3.11.5 cuda/12.2 gcc arrow/21.0.0

# ── Activate virtual environment ─────────────────────────────────────────────

cd "$project_dir"
source .venv/bin/activate

# ── Ensure hanabi-learning-environment is installed ──────────────────────────
# Our HLE adapter (env/hanabi/hanabi_env.py) wraps the standalone HLE package
# instead of OpenSpiel's C++ Hanabi extension (avoids BUILD_WITH_HANABI build).
pip install --quiet hanabi-learning-environment 2>/dev/null || true

# ── Print run info ───────────────────────────────────────────────────────────

echo "============================================"
echo " Full Hanabi RL — SLURM Job ${SLURM_JOB_ID}"
echo "============================================"
echo "  Game:         ${GAME}"
echo "  Model:        ${MODEL_ID}"
echo "  LoRA rank:    ${LORA_RANK}"
echo "  LR:           ${LR}"
echo "  GRPO Passes:  ${GRPO_PASSES}"
echo "  Collect eps:  ${COLLECT_EPISODES}"
echo "  Generations:  ${NUM_GENERATIONS}"
echo "  Max Seq Len:  ${MAX_SEQ_LEN}"
echo "  Profile:      ${PROFILE}"
echo "  Output dir:   ${output_dir}"
echo "  Node:         $(hostname)"
echo "  GPUs:         ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "  Python:       $(which python3)"
echo "  PyTorch CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "============================================"

# ── Pre-flight checks ────────────────────────────────────────────────────────

if [ -z "${HF_TOKEN}" ]; then
  echo "ERROR: No Hugging Face token found."
  echo "  Gemma 3 is a gated model — you must be authenticated."
  echo "  Steps:"
  echo "    1. Accept the license at https://huggingface.co/google/gemma-3-12b-it"
  echo "    2. Run: huggingface-cli login"
  echo "    3. Or set HF_TOKEN=hf_... in your environment before sbatch"
  exit 1
fi

mkdir -p "${output_dir}"
mkdir -p slurm/output

# ── Run training ─────────────────────────────────────────────────────────────

python3 trainer/gemma_rl_trainer.py \
  --rl_algorithm=grpo \
  --game="${GAME}" \
  --model_name="${MODEL_ID}" \
  --lora_rank="${LORA_RANK}" \
  --lora_alpha=$((LORA_RANK * 2)) \
  --lr="${LR}" \
  --grpo_passes="${GRPO_PASSES}" \
  --grpo_collect_episodes="${COLLECT_EPISODES}" \
  --grpo_num_generations="${NUM_GENERATIONS}" \
  --grpo_max_completion_length=32 \
  --eval_every=50 \
  --num_eval_episodes="${NUM_EVAL_EPISODES}" \
  --checkpoint_every=25 \
  --temperature=0.7 \
  --max_seq_len="${MAX_SEQ_LEN}" \
  --use_4bit \
  --output_dir="${output_dir}" \
  --log_every=5 \
  --log_episodes_every=5 \
  ${EXTRA_FLAGS}

echo "============================================"
echo " Training complete. Output: ${output_dir}"
echo "============================================"
