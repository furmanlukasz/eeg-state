#!/bin/bash
# =============================================================================
# Full Pipeline Orchestration Script for RunPod
# =============================================================================
# This script runs the complete experiment pipeline with both configurations:
# 1. Phase-only model (cos, sin)
# 2. Phase + Amplitude model (cos, sin, log_amp)
#
# Usage:
#   ./scripts/run_full_pipeline.sh              # Run everything sequentially
#   ./scripts/run_full_pipeline.sh --parallel   # Run phase/amplitude in parallel
#
# Environment variables:
#   EPOCHS=300          - Training epochs (default: 300)
#   HIDDEN_SIZE=128     - Encoder hidden size (default: 128)
#   BATCH_SIZE=64       - Training batch size (default: 64)
#   N_FOLDS=5           - Cross-validation folds (default: 5)
#   N_SEEDS=3           - Random seeds for experiments (default: 3)
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
EPOCHS=${EPOCHS:-300}
HIDDEN_SIZE=${HIDDEN_SIZE:-128}
BATCH_SIZE=${BATCH_SIZE:-64}
N_FOLDS=${N_FOLDS:-5}
N_SEEDS=${N_SEEDS:-3}
WANDB_ENABLED=${WANDB:-false}  # WandB disabled by default

# Use python directly (RunPod has torch pre-installed)
PY="python"

# Parse arguments
PARALLEL=false
for arg in "$@"; do
    case $arg in
        --parallel)
            PARALLEL=true
            shift
            ;;
    esac
done

echo -e "${BLUE}=============================================="
echo "EEG State Biomarkers - Full Pipeline"
echo "==============================================${NC}"
echo ""
echo "Configuration:"
echo "  EPOCHS: $EPOCHS"
echo "  HIDDEN_SIZE: $HIDDEN_SIZE"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  N_FOLDS: $N_FOLDS"
echo "  N_SEEDS: $N_SEEDS"
echo "  PARALLEL: $PARALLEL"
echo "  WANDB: $WANDB_ENABLED"
echo "  Python: $PY"
echo ""

# Create output directories
mkdir -p models results/phase_only results/phase_amplitude logs

# =============================================================================
# STEP 1: Training
# =============================================================================
train_model() {
    local name=$1
    local include_amp=$2
    local output_dir=$3

    echo -e "\n${YELLOW}[TRAINING] $name${NC}"

    $PY -m eeg_biomarkers.training.train \
        data=runpod \
        paths.data_dir=data \
        paths.model_dir=models \
        training.epochs=$EPOCHS \
        model.encoder.hidden_size=$HIDDEN_SIZE \
        training.batch_size=$BATCH_SIZE \
        model.phase.include_amplitude=$include_amp \
        logging.wandb.enabled=$WANDB_ENABLED \
        experiment.name="$name" \
        2>&1 | tee "logs/train_${name}.log"

    # Copy best model to named location
    if [ -f "models/best.pt" ]; then
        cp "models/best.pt" "models/${name}_best.pt"
        echo -e "${GREEN}Model saved: models/${name}_best.pt${NC}"
    fi
}

# =============================================================================
# STEP 2: Integration Experiment
# =============================================================================
run_integration() {
    local name=$1
    local checkpoint=$2
    local output_dir=$3

    echo -e "\n${YELLOW}[INTEGRATION] $name${NC}"

    $PY -m eeg_biomarkers.experiments.integration_experiment \
        --data-dir data \
        --output-dir "$output_dir" \
        --checkpoint "$checkpoint" \
        --n-folds $N_FOLDS \
        --n-seeds $N_SEEDS \
        2>&1 | tee "logs/integration_${name}.log"

    echo -e "${GREEN}Results saved: $output_dir${NC}"
}

# =============================================================================
# STEP 3: Sanity Checks
# =============================================================================
run_sanity_checks() {
    echo -e "\n${YELLOW}[SANITY CHECKS]${NC}"

    # Run sanity check pipeline if it exists
    if [ -f "scripts/sanity_check_pipeline.py" ]; then
        $PY scripts/sanity_check_pipeline.py 2>&1 | tee logs/sanity_check.log
    fi

    # Run amplitude sanity check if it exists
    if [ -f "scripts/sanity_check_with_amplitude.py" ]; then
        $PY scripts/sanity_check_with_amplitude.py 2>&1 | tee logs/sanity_check_amplitude.log
    fi
}

# =============================================================================
# Main Execution
# =============================================================================

if [ "$PARALLEL" = true ]; then
    echo -e "\n${BLUE}Running PARALLEL training...${NC}"

    # Train both models in parallel using background processes
    train_model "phase_only" "false" "results/phase_only" &
    PID1=$!

    train_model "phase_amplitude" "true" "results/phase_amplitude" &
    PID2=$!

    # Wait for both to complete
    echo "Waiting for training jobs (PIDs: $PID1, $PID2)..."
    wait $PID1
    STATUS1=$?
    wait $PID2
    STATUS2=$?

    if [ $STATUS1 -ne 0 ] || [ $STATUS2 -ne 0 ]; then
        echo -e "${RED}Training failed!${NC}"
        exit 1
    fi

    echo -e "\n${BLUE}Running PARALLEL integration experiments...${NC}"

    # Run integration experiments in parallel
    run_integration "phase_only" "models/phase_only_best.pt" "results/phase_only" &
    PID1=$!

    run_integration "phase_amplitude" "models/phase_amplitude_best.pt" "results/phase_amplitude" &
    PID2=$!

    wait $PID1
    wait $PID2

else
    echo -e "\n${BLUE}Running SEQUENTIAL training...${NC}"

    # Phase-only model
    train_model "phase_only" "false" "results/phase_only"
    run_integration "phase_only" "models/phase_only_best.pt" "results/phase_only"

    # Phase + Amplitude model
    train_model "phase_amplitude" "true" "results/phase_amplitude"
    run_integration "phase_amplitude" "models/phase_amplitude_best.pt" "results/phase_amplitude"
fi

# Run sanity checks at the end
run_sanity_checks

# =============================================================================
# Summary
# =============================================================================
echo -e "\n${GREEN}=============================================="
echo "Pipeline Complete!"
echo "==============================================${NC}"
echo ""
echo "Models:"
echo "  - models/phase_only_best.pt"
echo "  - models/phase_amplitude_best.pt"
echo ""
echo "Results:"
echo "  - results/phase_only/"
echo "  - results/phase_amplitude/"
echo ""
echo "Logs:"
echo "  - logs/train_*.log"
echo "  - logs/integration_*.log"
echo ""
echo "Compare results with:"
echo "  cat results/phase_only/summary.json"
echo "  cat results/phase_amplitude/summary.json"
