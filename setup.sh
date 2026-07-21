#!/bin/bash
# TeamGamesRL — environment setup
# Usage: source setup.sh  (or: bash setup.sh && source .venv/bin/activate)

set -e


echo "🚀 Starting TeamGamesRL setup..."


# ── 1. Load system modules ──────────────────────────────────────────────────
echo "📦 Loading system modules..."
module load python/3.11.5 cuda/12.2 gcc arrow/21.0.0 rust

# OpenSpiel requires clang++ (>= 17) with C++20 support to build from source.
# Try loading an LLVM / Clang module if available on this cluster.
echo "  → Checking for Clang (required by open-spiel build)..."
if ! command -v clang++ &>/dev/null; then
    # Try common module names across ComputeCanada / internal clusters.
    for MOD in llvm/17 llvm/18 llvm clang/17 clang/18 clang; do
        if module is-avail "$MOD" 2>/dev/null; then
            echo "  → Loading module: $MOD"
            module load "$MOD"
            break
        fi
    done
fi

# If clang++ is still not found, try using GCC as the C++ compiler.
# open-spiel's CMake build *can* use g++ if CXX is set before pip runs.
if ! command -v clang++ &>/dev/null; then
    echo "  ⚠️  clang++ not found — falling back to g++ via CXX override"
    export CXX="$(command -v g++)"
    export CC="$(command -v gcc)"

    # If g++ still doesn't exist, install clang into the virtualenv below.
    if [ -z "$CXX" ]; then
        echo "  ❌ Neither clang++ nor g++ found. Please install a C++20 compiler."
        echo "     On Debian/Ubuntu: sudo apt install clang-17"
        echo "     On ComputeCanada: module load llvm"
        exit 1
    fi
fi

echo "  → C++ compiler: $(command -v clang++ 2>/dev/null || echo "$CXX")"
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

# Install open-spiel separately — it has a heavy C++ build step.
# Try a pre-built wheel first; fall back to source build.
echo "  → Installing open-spiel..."
if ! pip install open-spiel --prefer-binary --only-binary open-spiel 2>/dev/null; then
    echo "  → No pre-built wheel available; building open-spiel from source..."
    echo "  → Limiting parallel jobs to avoid OOM during compilation..."

    # open-spiel's setup.py uses Python's os.cpu_count() (not shell nproc)
    # and passes --parallel <N> directly to cmake. On 48-core nodes this
    # spawns 48 concurrent g++ processes (~2 GB each for pybind11 TUs),
    # which OOM-kills the build. The --parallel CLI flag overrides the
    # CMAKE_BUILD_PARALLEL_LEVEL env var, so we MUST intercept the cmake
    # command itself.
    #
    # Strategy: create a cmake wrapper that caps --parallel to 2 jobs.
    NPROC_LIMIT=2
    WRAPPER_DIR=$(mktemp -d)
    REAL_CMAKE="$(command -v cmake)"

    cat > "$WRAPPER_DIR/cmake" << 'WRAPPER_EOF'
#!/bin/bash
# cmake wrapper — caps --parallel to avoid OOM during open-spiel build.
LIMIT=2
NEWARGS=()
CAP_NEXT=false
for arg in "$@"; do
    if $CAP_NEXT; then
        # Replace the parallel count with our limit
        NEWARGS+=("$LIMIT")
        CAP_NEXT=false
    elif [ "$arg" = "--parallel" ]; then
        NEWARGS+=("$arg")
        CAP_NEXT=true
    else
        NEWARGS+=("$arg")
    fi
done
exec REAL_CMAKE_PLACEHOLDER "${NEWARGS[@]}"
WRAPPER_EOF

    # Patch the placeholder with the actual cmake path.
    sed -i "s|REAL_CMAKE_PLACEHOLDER|${REAL_CMAKE}|g" "$WRAPPER_DIR/cmake"
    chmod +x "$WRAPPER_DIR/cmake"

    export PATH="$WRAPPER_DIR:$PATH"
    export CMAKE_BUILD_PARALLEL_LEVEL=$NPROC_LIMIT
    export MAKEFLAGS="-j$NPROC_LIMIT"

    echo "  → Build parallelism capped at $NPROC_LIMIT jobs (cmake wrapper active)"
    echo "  → Real cmake: $REAL_CMAKE"
    pip install open-spiel

    # Restore defaults.
    export PATH="${PATH#"$WRAPPER_DIR:"}"
    unset CMAKE_BUILD_PARALLEL_LEVEL
    unset MAKEFLAGS
    rm -rf "$WRAPPER_DIR"
fi

# Install remaining deps (skip open-spiel since it's already installed).
echo "  → Installing remaining dependencies..."
pip install -r requirements.txt \
    --find-links https://pypi.org/simple/ \
    --prefer-binary
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
