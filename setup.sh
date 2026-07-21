#!/bin/bash
# TeamGamesRL — environment setup
# Usage: source setup.sh  (or: bash setup.sh && source .venv/bin/activate)

set -e


echo "🚀 Starting TeamGamesRL setup..."


# ── 1. Load system modules ──────────────────────────────────────────────────
echo "📦 Loading system modules..."
module load python/3.11.5 cuda/12.2 gcc arrow/21.0.0 rust
echo "✅ System modules loaded"


# ── 2. Create & activate virtual environment ─────────────────────────────────
echo "🌍 Creating Python virtual environment..."
virtualenv --no-download .venv --prompt TeamGamesRL
echo "  → Activating virtual environment..."
source .venv/bin/activate
echo "✅ Virtual environment created and activated"


# ── 3. Install Python dependencies ──────────────────────────────────────────
echo "📚 Installing dependencies..."
echo "  → This may take a few minutes..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --find-links https://pypi.org/simple/ --prefer-binary
echo "✅ All dependencies installed"


# ── 4. Verify critical imports ───────────────────────────────────────────────
echo "🔍 Verifying key packages..."
python -c "
import open_spiel; print(f'  open_spiel  {open_spiel.__file__}')
import torch;      print(f'  torch       {torch.__version__}  (CUDA: {torch.cuda.is_available()})')
import transformers; print(f'  transformers {transformers.__version__}')
import trl;        print(f'  trl         {trl.__version__}')
import peft;       print(f'  peft        {peft.__version__}')
"
echo "✅ All critical packages verified"


# ── 5. Optional: Hugging Face login ─────────────────────────────────────────
echo ""
echo "🔑 Gemma 2B is a gated model — you need a Hugging Face token with access."
read -p "Log into Hugging Face now? (y/n): "
if [[ $REPLY =~ ^[Yy]$ ]]; then
    huggingface-cli login
    echo "✅ Logged into Hugging Face"
else
    echo "⏭️  Skipping Hugging Face login."
    echo "   You can log in later with: huggingface-cli login"
fi


echo ""
echo "🎉 Setup complete!"
echo "💡 Activate the environment with: source .venv/bin/activate"
echo "🚂 Start training with:           python gemma_rl_trainer.py --game=tiny_hanabi"
