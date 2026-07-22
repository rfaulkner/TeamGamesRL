#!/bin/bash
#SBATCH --job-name=teamgamesrl
#SBATCH --account=aip-rgrosse
#SBATCH --output=slurm/output/%j_%x.out
#SBATCH --error=slurm/output/%j_%x.err

#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# ============================================================================
# TeamGamesRL — SLURM submission script
#
# Usage:
#   sbatch run_rl.slurm                              # all defaults
#   sbatch run_rl.slurm tiny_hanabi                   # specify game
#   sbatch run_rl.slurm hanabi google/gemma-2-2b 32   # game, model, lora_rank
#   sbatch run_rl.slurm tiny_hanabi google/gemma-2-2b 16 1e-4 500
#
# Positional args:
#   $1 = game           (default: tiny_hanabi)
#   $2 = model_id       (default: google/gemma-2-2b)
#   $3 = lora_rank      (default: 16)
#   $4 = learning_rate  (default: 1e-4)
#   $5 = num_episodes   (default: 500)
# ============================================================================

set -euo pipefail

# ── Parse arguments with defaults ────────────────────────────────────────────

GAME="${1:-tiny_hanabi}"
MODEL_ID="${2:-google/gemma-2-2b}"
LORA_RANK="${3:-16}"
LR="${4:-1e-4}"
NUM_EPISODES="${5:-500}"

# ── Derived settings ─────────────────────────────────────────────────────────

project_dir="/home/$USER/projects/aip-rgrosse/$USER/TeamGamesRL"
output_dir="/scratch/$USER/teamgamesrl/${GAME}_lr${LR}_rank${LORA_RANK}_ep${NUM_EPISODES}_${SLURM_JOB_ID}"

export HF_HOME="/scratch/$USER/hf_cache"
export WANDB_DISABLED=true  # Set to "false" and add --use_wandb below to enable
export PYTHONUNBUFFERED=1   # Flush all Python output immediately to SLURM logs

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

echo "============================================"
echo " TeamGamesRL — SLURM Job ${SLURM_JOB_ID}"
echo "============================================"
echo "  Game:         ${GAME}"
echo "  Model:        ${MODEL_ID}"
echo "  LoRA rank:    ${LORA_RANK}"
echo "  LR:           ${LR}"
echo "  Episodes:     ${NUM_EPISODES}"
echo "  Output dir:   ${output_dir}"
echo "  Node:         $(hostname)"
echo "  GPUs:         ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "  Python:       $(which python3)"
echo "  PyTorch CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "============================================"

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

python3 gemma_rl_trainer.py \
  --game="${GAME}" \
  --model_name="${MODEL_ID}" \
  --lora_rank="${LORA_RANK}" \
  --lora_alpha=$((LORA_RANK * 2)) \
  --lr="${LR}" \
  --num_episodes="${NUM_EPISODES}" \
  --eval_every=50 \
  --num_eval_episodes=10 \
  --checkpoint_every=100 \
  --temperature=0.8 \
  --max_seq_len=512 \
  --use_4bit \
  --output_dir="${output_dir}" \
  --log_every=10

echo "============================================"
echo " Training complete. Output: ${output_dir}"
echo "============================================"

