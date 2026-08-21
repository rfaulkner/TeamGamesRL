#!/bin/bash
#SBATCH --job-name=teamgamesrl_grpo
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
# TeamGamesRL — GRPO SLURM submission script
#
# Usage:
#   sbatch scripts/run_grpo.sh                              # all defaults (tiny_hanabi, 30 passes)
#   sbatch scripts/run_grpo.sh tiny_hanabi                  # specify game
#   sbatch scripts/run_grpo.sh hanabi google/gemma-2-2b 32  # game, model, lora_rank
#   sbatch scripts/run_grpo.sh tiny_hanabi google/gemma-2-2b 16 3e-5 30 50
#
# Positional args:
#   $1 = game                  (default: tiny_hanabi)
#   $2 = model_id              (default: google/gemma-2-2b)
#   $3 = lora_rank             (default: 16)
#   $4 = learning_rate         (default: 3e-5)
#   $5 = grpo_passes           (default: 30)
#   $6 = grpo_collect_episodes (default: 50)
# ============================================================================

set -euo pipefail

# ── Parse arguments with defaults ────────────────────────────────────────────

GAME="${1:-tiny_hanabi}"
MODEL_ID="${2:-google/gemma-2-2b}"
LORA_RANK="${3:-16}"
LR="${4:-3e-5}"
GRPO_PASSES="${5:-30}"
GRPO_COLLECT_EPISODES="${6:-50}"

# ── Derived settings ─────────────────────────────────────────────────────────

project_dir="/home/$USER/projects/aip-rgrosse/$USER/TeamGamesRL"
output_dir="/scratch/$USER/teamgamesrl/${GAME}_grpo_lr${LR}_rank${LORA_RANK}_passes${GRPO_PASSES}_${SLURM_JOB_ID}"

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

echo "============================================"
echo " TeamGamesRL — GRPO SLURM Job ${SLURM_JOB_ID}"
echo "============================================"
echo "  Algorithm:    GRPO"
echo "  Game:         ${GAME}"
echo "  Model:        ${MODEL_ID}"
echo "  LoRA rank:    ${LORA_RANK}"
echo "  LR:           ${LR}"
echo "  GRPO Passes:  ${GRPO_PASSES}"
echo "  Collect Eps:  ${GRPO_COLLECT_EPISODES}"
echo "  Output dir:   ${output_dir}"
echo "  Node:         $(hostname)"
echo "  GPUs:         ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "  Python:       $(which python3)"
echo "  PyTorch CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "============================================"

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

python3 trainer/gemma_rl_trainer.py \
  --rl_algorithm=grpo \
  --game="${GAME}" \
  --model_name="${MODEL_ID}" \
  --lora_rank="${LORA_RANK}" \
  --lora_alpha=$((LORA_RANK * 2)) \
  --lr="${LR}" \
  --grpo_passes="${GRPO_PASSES}" \
  --grpo_collect_episodes="${GRPO_COLLECT_EPISODES}" \
  --eval_every=50 \
  --num_eval_episodes=10 \
  --checkpoint_every=100 \
  --temperature=0.8 \
  --max_seq_len=512 \
  --use_4bit \
  --output_dir="${output_dir}" \
  --log_every=10

echo "============================================"
echo " GRPO Training complete. Output: ${output_dir}"
echo "============================================"
