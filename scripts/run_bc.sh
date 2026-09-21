#!/bin/bash
#SBATCH --job-name=hanabi-bc
#SBATCH --account=aip-rgrosse
#SBATCH --output=slurm/output/%j_%x.out
#SBATCH --error=slurm/output/%j_%x.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

# ============================================================================
# Hanabi Behavioral Cloning (BC) Warm-Start Training Script
#
# 1. Generates belief-search expert demonstration data (if needed).
# 2. Trains LoRA adapter on Gemma via Supervised Fine-Tuning (SFT).
# 3. Emits a pre-trained adapter ready for warm-starting GRPO/RL.
# ============================================================================

set -euo pipefail

MODEL_ID="google/gemma-2-2b" # "google/gemma-3-12b-it"
NUM_GAMES=100
N_WORLDS=3
EPOCHS=3
BATCH_SIZE=4
GRAD_ACCUM=4
LR="1e-4"
REASONING=""
DATA_DIR=""
FORCE_REGEN=false

for arg in "$@"; do
  case "$arg" in
    --model=*) MODEL_ID="${arg#*=}" ;;
    --num_games=*) NUM_GAMES="${arg#*=}" ;;
    --n_worlds=*) N_WORLDS="${arg#*=}" ;;
    --epochs=*) EPOCHS="${arg#*=}" ;;
    --batch_size=*) BATCH_SIZE="${arg#*=}" ;;
    --lr=*) LR="${arg#*=}" ;;
    --reasoning) REASONING="--reasoning" ;;
    --data_dir=*) DATA_DIR="${arg#*=}" ;;
    --force_regen) FORCE_REGEN=true ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

project_dir="/home/$USER/projects/aip-rgrosse/$USER/TeamGamesRL"
if [ -n "${DATA_DIR}" ]; then
  data_dir="${DATA_DIR}"
elif [ -n "${REASONING}" ]; then
  data_dir="${project_dir}/data/bc_hanabi_reasoning"
else
  data_dir="${project_dir}/data/bc_hanabi"
fi
MODEL_TAG=$(echo "$MODEL_ID" | tr '/' '_')
output_dir="${project_dir}/checkpoints/bc_${MODEL_TAG}_${SLURM_JOB_ID}"

export HF_HOME="/scratch/$USER/hf_cache"
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export PYTHONPATH="${project_dir}:${PYTHONPATH:-.}"
export HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null || true)}"

# ── Load system modules ─────────────────────────────────────────────────────
module load python/3.11.5 cuda/12.2 gcc arrow/21.0.0

cd "$project_dir"
source .venv/bin/activate
pip install --quiet hanabi-learning-environment 2>/dev/null || true

mkdir -p slurm/output
mkdir -p "${data_dir}"
mkdir -p "${output_dir}"

echo "============================================"
echo " Hanabi BC Warm-Start — SLURM Job ${SLURM_JOB_ID}"
echo "============================================"
echo "  Model:       ${MODEL_ID}"
echo "  Num games:   ${NUM_GAMES}"
echo "  Output dir:  ${output_dir}"
echo "============================================"

# Step 1: Generate BC dataset if train.jsonl does not exist or force_regen is true
if [ ! -f "${data_dir}/train.jsonl" ] || [ "${FORCE_REGEN}" = "true" ]; then
  echo "[Step 1] Generating BC demonstration data from SafeBeliefLookaheadPlayer..."
  python3 data/generate_bc_data.py \
    --num_games="${NUM_GAMES}" \
    --n_worlds="${N_WORLDS}" \
    --output_dir="${data_dir}" \
    ${REASONING}
else
  echo "[Step 1] Existing BC data found in ${data_dir}. Reusing dataset (pass --force_regen to overwrite)."
fi

# Step 2: Run Supervised Fine-Tuning
echo "[Step 2] Training BC LoRA model on ${MODEL_ID}..."
python3 train_bc.py \
  --train_file="${data_dir}/train.jsonl" \
  --val_file="${data_dir}/val.jsonl" \
  --model_name="${MODEL_ID}" \
  --epochs="${EPOCHS}" \
  --batch_size="${BATCH_SIZE}" \
  --gradient_accumulation_steps="${GRAD_ACCUM}" \
  --lr="${LR}" \
  --output_dir="${output_dir}"

echo "============================================"
echo " BC Training complete!"
echo " Best adapter saved to: ${output_dir}/best_adapter"
echo ""
echo " To warm-start GRPO with this adapter, launch:"
echo "   sbatch run_hanabi_full.sh --initial_lora_checkpoint=${output_dir}/best_adapter"
echo "============================================"
