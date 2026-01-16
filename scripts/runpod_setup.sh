#!/bin/bash
# RunPod Setup Script for EEG State Biomarkers
#
# RunPod comes with PyTorch 2.4 pre-installed, so we just need to:
# 1. Install UV
# 2. Clone/pull repo
# 3. Install additional dependencies (not torch)
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/furmanlukasz/eeg-state/main/scripts/runpod_setup.sh | bash
#
# Or if repo is already cloned:
#   bash scripts/runpod_setup.sh

set -e  # Exit on error

echo "=============================================="
echo "EEG State Biomarkers - RunPod Setup"
echo "=============================================="
echo "Note: Using pre-installed PyTorch from RunPod"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration - EDIT THESE
REPO_URL="${REPO_URL:-https://github.com/furmanlukasz/eeg-state.git}"
REPO_DIR="${REPO_DIR:-/workspace/eeg-state}"
# Default to /workspace/data on RunPod (standard location)
DATA_SOURCE="${DATA_SOURCE:-/workspace/data}"

# ---------------------------------------------
# 1. System Info
# ---------------------------------------------
echo -e "\n${YELLOW}[1/7] System Information${NC}"
echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo "Python: $(python3 --version 2>/dev/null || echo 'Not found')"
echo "CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo 'Not found')"

# Check GPU
echo -e "\n${YELLOW}GPU Information:${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo -e "${RED}nvidia-smi not found - GPU may not be available${NC}"
fi

# ---------------------------------------------
# 2. Install UV (fast Python package manager)
# ---------------------------------------------
echo -e "\n${YELLOW}[2/7] Installing UV${NC}"
if command -v uv &> /dev/null; then
    echo "UV already installed: $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
    echo -e "${GREEN}UV installed: $(uv --version)${NC}"
fi

# ---------------------------------------------
# 3. Clone or Pull Repository
# ---------------------------------------------
echo -e "\n${YELLOW}[3/7] Setting up repository${NC}"
if [ -d "$REPO_DIR" ]; then
    echo "Repository exists, pulling latest changes..."
    cd "$REPO_DIR"
    git fetch origin
    git reset --hard origin/main
    echo -e "${GREEN}Repository updated${NC}"
else
    echo "Cloning repository..."
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
    echo -e "${GREEN}Repository cloned${NC}"
fi

# ---------------------------------------------
# 4. Install Python Dependencies
# ---------------------------------------------
echo -e "\n${YELLOW}[4/7] Installing Python dependencies${NC}"
cd "$REPO_DIR"

# Check if we should use system Python (RunPod has torch pre-installed)
# or create a new venv
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "Using system Python with pre-installed PyTorch..."
    # Install additional deps with pip (skip torch since it's already there)
    pip install --quiet mne>=1.5 xgboost>=2.0 catboost>=1.2 lightgbm>=4.0 \
        umap-learn>=0.5 hdbscan>=0.8 numba>=0.58 hydra-core>=1.3 omegaconf>=2.3 \
        pandas>=2.0 matplotlib>=3.7 plotly>=5.15 seaborn>=0.12 \
        rich>=13.0 tqdm>=4.65 pytest>=7.0 pytest-cov>=4.0

    # Install the package in editable mode
    pip install -e .

    # Set flag to use python directly instead of uv run
    export USE_SYSTEM_PYTHON=true
    echo 'export USE_SYSTEM_PYTHON=true' >> ~/.bashrc
    # Also add to current session
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "Creating new virtual environment with uv..."
    uv sync --extra dev
    USE_SYSTEM_PYTHON=false
fi

echo -e "${GREEN}Dependencies installed${NC}"

# ---------------------------------------------
# 5. Verify PyTorch CUDA
# ---------------------------------------------
echo -e "\n${YELLOW}[5/7] Verifying PyTorch CUDA${NC}"

# Use appropriate python command
if [ "$USE_SYSTEM_PYTHON" = true ]; then
    PY_CMD="python"
else
    PY_CMD="uv run python"
fi

$PY_CMD -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name}')
        print(f'    Memory: {props.total_memory / 1024**3:.1f} GB')
        print(f'    Compute capability: {props.major}.{props.minor}')
else:
    print('WARNING: CUDA not available!')
"

# ---------------------------------------------
# 6. Setup Data Directory
# ---------------------------------------------
echo -e "\n${YELLOW}[6/7] Setting up data directory${NC}"
DATA_DIR="$REPO_DIR/data"

# Check if we're on RunPod and /workspace/data exists
if [ -d "/workspace/data" ]; then
    echo "Detected RunPod environment with data at /workspace/data"
    DATA_SOURCE="/workspace/data"
fi

if [ -n "$DATA_SOURCE" ] && [ -d "$DATA_SOURCE" ]; then
    # Link to external data source (removes existing dir/link first)
    if [ -L "$DATA_DIR" ]; then
        rm "$DATA_DIR"
    elif [ -d "$DATA_DIR" ]; then
        rm -rf "$DATA_DIR"
    fi
    ln -sf "$DATA_SOURCE" "$DATA_DIR"
    echo -e "${GREEN}Data symlinked: $DATA_DIR -> $DATA_SOURCE${NC}"
elif [ -d "$DATA_DIR" ]; then
    echo "Data directory exists at $DATA_DIR"
else
    mkdir -p "$DATA_DIR"
    echo -e "${YELLOW}Data directory created at $DATA_DIR${NC}"
    echo "  You need to copy/mount your EEG data here"
    echo "  Expected structure:"
    echo "    data/HID/subject_*/  (Healthy controls)"
    echo "    data/MCI/subject_*/  (MCI patients)"
fi

# Count data files if they exist (follow symlinks)
if [ -d "$DATA_DIR/HID" ] || [ -d "$DATA_DIR/MCI" ]; then
    HC_COUNT=$(find -L "$DATA_DIR/HID" -name "*_eeg.fif" 2>/dev/null | wc -l)
    MCI_COUNT=$(find -L "$DATA_DIR/MCI" -name "*_eeg.fif" 2>/dev/null | wc -l)
    echo "  Found: $HC_COUNT HC files, $MCI_COUNT MCI files"
fi

# ---------------------------------------------
# 7. Create convenience scripts
# ---------------------------------------------
echo -e "\n${YELLOW}[7/7] Creating convenience scripts${NC}"

# Training script - use python directly since RunPod has torch pre-installed
cat > "$REPO_DIR/run_train.sh" << 'TRAIN_EOF'
#!/bin/bash
# Train the autoencoder on full dataset
cd /workspace/eeg-state

# Use runpod config for proper data path handling
python -m eeg_biomarkers.training.train \
    data=runpod \
    paths.data_dir=data \
    training.epochs=${EPOCHS:-300} \
    model.encoder.hidden_size=${HIDDEN_SIZE:-128} \
    training.batch_size=${BATCH_SIZE:-64} \
    "$@"
TRAIN_EOF
chmod +x "$REPO_DIR/run_train.sh"

# Integration experiment script
cat > "$REPO_DIR/run_experiment.sh" << 'EXP_EOF'
#!/bin/bash
# Run integration experiment
cd /workspace/eeg-state

python -m eeg_biomarkers.experiments.integration_experiment \
    --data-dir data \
    --output-dir results/integration \
    --checkpoint ${CHECKPOINT:-models/best.pt} \
    --n-folds ${N_FOLDS:-5} \
    --n-seeds ${N_SEEDS:-3} \
    "$@"
EXP_EOF
chmod +x "$REPO_DIR/run_experiment.sh"

# Test script
cat > "$REPO_DIR/run_tests.sh" << 'TEST_EOF'
#!/bin/bash
# Run tests
cd /workspace/eeg-state
pytest tests/ -v --tb=short "$@"
TEST_EOF
chmod +x "$REPO_DIR/run_tests.sh"

echo -e "${GREEN}Convenience scripts created:${NC}"
echo "  ./run_train.sh       - Train autoencoder (phase only by default)"
echo "  ./run_experiment.sh  - Run integration experiment"
echo "  ./run_tests.sh       - Run tests"
echo ""
echo "Full pipeline script available at: scripts/run_full_pipeline.sh"

# ---------------------------------------------
# Done!
# ---------------------------------------------
echo -e "\n${GREEN}=============================================="
echo "Setup Complete!"
echo "==============================================${NC}"
echo ""
echo "Quick Start:"
echo "  cd $REPO_DIR"
echo ""
echo "  # Run tests first"
echo "  ./run_tests.sh"
echo ""
echo -e "${YELLOW}=== TRAINING OPTIONS ===${NC}"
echo ""
echo "  # Option 1: Train phase-only model (default, no amplitude)"
echo "  EPOCHS=300 ./run_train.sh"
echo ""
echo "  # Option 2: Train with amplitude"
echo "  EPOCHS=300 ./run_train.sh model.phase.include_amplitude=true"
echo ""
echo "  # Option 3: Full parallel pipeline (both models + experiments)"
echo "  EPOCHS=300 bash scripts/run_full_pipeline.sh --parallel"
echo ""
echo -e "${YELLOW}=== EXPERIMENT OPTIONS ===${NC}"
echo ""
echo "  # Run integration experiment on trained model"
echo "  CHECKPOINT=models/best.pt ./run_experiment.sh"
echo ""
echo -e "${YELLOW}=== ADVANCED ===${NC}"
echo ""
echo "  # Direct Python commands"
echo "  python -m eeg_biomarkers.training.train --help"
echo "  python -m eeg_biomarkers.experiments.integration_experiment --help"
