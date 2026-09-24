#!/bin/bash
#SBATCH --job-name=hanabi-rl
#SBATCH --account=aip-rgrosse
#SBATCH --output=slurm/output/%j_%x.out
#SBATCH --error=slurm/output/%j_%x.err

#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# ============================================================================
# Hanabi RL — SLURM submission script
#
# Trains a Gemma model with GRPO on Hanabi variants.
#
# Usage:
#   sbatch run_hanabi.sh                              # tiny hanabi (default)
#   sbatch run_hanabi.sh --game=hanabi --max_seq_len=2048  # full hanabi
#   sbatch run_hanabi.sh --grpo_passes=50             # override one flag
#   sbatch run_hanabi.sh --lr=1e-4                    # override several
#   sbatch run_hanabi.sh --model=google/gemma-2-2b --lora_rank=32
#
# Flags (all optional, order doesn't matter):
#   --game=NAME         Game variant           (default: tiny_hanabi)
#   --model=ID          HF model identifier    (default: google/gemma-2-2b)
#   --lora_rank=R       LoRA rank              (default: 16)
#   --lr=RATE           Learning rate          (default: 3e-5)
#   --num_episodes=N    Training episodes      (default: 500)
#   --grpo_passes=P     GRPO passes            (default: 30)
#   --max_seq_len=L     Max sequence length    (default: 512)
# ============================================================================

set -euo pipefail

# Save original CLI invocation for logging in .out file
CLI_ARGS=("$@")

# ── Parse arguments with defaults ────────────────────────────────────────────

GAME="tiny_hanabi"
MODEL_ID="google/gemma-2-2b"
LORA_RANK=16
LR="3e-5"
NUM_EPISODES=500
GRPO_PASSES=30
MAX_SEQ_LEN=512
EXTRA_FLAGS=""

for arg in "$@"; do
  case "$arg" in
    --game=*)         GAME="${arg#*=}" ;;
    --model=*)        MODEL_ID="${arg#*=}" ;;
    --lora_rank=*)    LORA_RANK="${arg#*=}" ;;
    --lr=*)           LR="${arg#*=}" ;;
    --num_episodes=*) NUM_EPISODES="${arg#*=}" ;;
    --grpo_passes=*)  GRPO_PASSES="${arg#*=}" ;;
    --max_seq_len=*)  MAX_SEQ_LEN="${arg#*=}" ;;
    --help|-h)
      echo "Usage: sbatch run_hanabi.sh [--game=G] [--model=M] [--lora_rank=R] [--lr=L] [--num_episodes=N] [--grpo_passes=P] [--max_seq_len=L] [--extra_trainer_flags...]"
      exit 0 ;;
    --*) EXTRA_FLAGS="${EXTRA_FLAGS} ${arg}" ;;
    *) echo "Unknown flag: $arg (try --help)"; exit 1 ;;
  esac
done

# ── Derived settings ─────────────────────────────────────────────────────────

project_dir="/home/$USER/projects/aip-rgrosse/$USER/TeamGamesRL"
output_dir="/scratch/$USER/teamgamesrl/${GAME}_lr${LR}_rank${LORA_RANK}_passes${GRPO_PASSES}_${SLURM_JOB_ID}"

export HF_HOME="/scratch/$USER/hf_cache"
export WANDB_DISABLED=true  # Set to "false" and add --use_wandb below to enable
export PYTHONUNBUFFERED=1   # Flush all Python output immediately to SLURM logs
export PYTHONPATH="${project_dir}:${PYTHONPATH:-.}"

# ── Hugging Face auth (Gemma is a gated model) ──────────────────────────────
# The token is needed to download gated models like google/gemma-2-2b.
# Run `huggingface-cli login` once interactively before submitting jobs.
# The token is saved to ~/.cache/huggingface/token by default.
export HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null || true)}"

# ── Load system modules ─────────────────────────────────────────────────────

module load python/3.11.5 cuda/12.2 gcc arrow/21.0.0

# ── Activate virtual environment ─────────────────────────────────────────────

cd "$project_dir"
source .venv/bin/activate

# ── Print run info ───────────────────────────────────────────────────────────

echo "=============================================================================="
echo " Hanabi RL — SLURM Job ${SLURM_JOB_ID:-N/A} (${SLURM_JOB_NAME:-hanabi-rl})"
echo " Started:      $(date)"
echo " Node:         $(hostname)"
echo " GPUs:         ${CUDA_VISIBLE_DEVICES:-N/A}"
echo " Invocation:   $0 ${CLI_ARGS[*]:-<none>}"
echo "=============================================================================="
echo "  Game:                        ${GAME}"
echo "  Model:                       ${MODEL_ID}"
echo "  LoRA rank:                   ${LORA_RANK} (alpha=$((LORA_RANK * 2)))"
echo "  Learning Rate (LR):          ${LR}"
echo "  Episodes:                    ${NUM_EPISODES}"
echo "  GRPO Passes:                 ${GRPO_PASSES}"
echo "  Max Seq Len:                 ${MAX_SEQ_LEN}"
echo "  Extra Flags:                 ${EXTRA_FLAGS:-<none>}"
echo "  Output Dir:                  ${output_dir}"
echo "  Python:                      $(which python3)"
echo "  PyTorch CUDA:                $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "=============================================================================="

# ── Create output directories ────────────────────────────────────────────────

# ── Pre-flight checks ────────────────────────────────────────────────────────

if [ -z "${HF_TOKEN}" ]; then
  echo "ERROR: No Hugging Face token found."
  echo "  Gemma is a gated model — you must be authenticated."
  echo "  Steps:"
  echo "    1. Accept the license at https://huggingface.co/google/gemma-2-2b"
  echo "    2. Run: huggingface-cli login"
  echo "    3. Or set HF_TOKEN=hf_... in your environment before sbatch"
  exit 1
fi

mkdir -p "${output_dir}"
mkdir -p slurm/output

# ── Run training ─────────────────────────────────────────────────────────────

echo "Launching training process: python3 trainer/gemma_rl_trainer.py ..."
python3 trainer/gemma_rl_trainer.py \
  --rl_algorithm=grpo \
  --game="${GAME}" \
  --model_name="${MODEL_ID}" \
  --lora_rank="${LORA_RANK}" \
  --lora_alpha=$((LORA_RANK * 2)) \
  --lr="${LR}" \
  --num_episodes="${NUM_EPISODES}" \
  --grpo_passes="${GRPO_PASSES}" \
  --eval_every=50 \
  --num_eval_episodes=10 \
  --checkpoint_every=100 \
  --temperature=0.8 \
  --max_seq_len="${MAX_SEQ_LEN}" \
  --use_4bit \
  --output_dir="${output_dir}" \
  --log_every=10 \
  ${EXTRA_FLAGS}

echo "============================================"
echo " Training complete. Output: ${output_dir}"
echo "============================================"
