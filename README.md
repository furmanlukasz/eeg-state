# EEG State Biomarkers

**State-aware dynamical biomarkers of MCI from resting-state EEG**

> Work in progress - PhD thesis project

## Overview

This package implements a ConvLSTM autoencoder framework for discovering latent brain states in resting-state EEG and computing state-conditioned dynamical biomarkers for Mild Cognitive Impairment (MCI) classification.

### Core Hypothesis

Resting EEG is a mixture of latent brain states; MCI alters the dynamics within and transitions between these states. Reliable biomarkers emerge only when features are computed within stable regimes.

## Installation

```bash
# With UV (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"
```

## Quick Start

### Training

```bash
# Train autoencoder with default config
uv run python -m eeg_biomarkers.training.train

# Train on full dataset
uv run python -m eeg_biomarkers.training.train \
    data=full \
    training.epochs=300 \
    model.encoder.hidden_size=128
```

### Integration Experiment

```bash
uv run python -m eeg_biomarkers.experiments.integration_experiment \
    --data-dir data \
    --output-dir results/integration \
    --checkpoint models/best.pt \
    --n-folds 5 \
    --n-seeds 3
```

### Tests

```bash
uv run pytest tests/ -v

# On macOS/Apple Silicon, some tests may segfault due to torch/MPS issues
# Tests run cleanly on Linux (RunPod/CUDA)
```

## Data

Expected data structure:
```
data/
├── HID/           # Healthy controls
│   └── subject_*/
│       └── *_eeg.fif
└── MCI/           # MCI patients
    └── subject_*/
        └── *_eeg.fif
```

Data files are not included in this repository (too large). Copy or symlink your data directory.

## Project Structure

```
eeg-state-biomarkers/
├── src/eeg_biomarkers/
│   ├── models/          # ConvLSTM autoencoder
│   ├── data/            # Data loading & preprocessing
│   ├── training/        # Training infrastructure
│   ├── analysis/        # RQA, state discovery, classification
│   └── experiments/     # Integration experiment
├── configs/             # Hydra YAML configs
├── tests/               # pytest test suite
└── CLAUDE.md            # Development context
```

## Key Features

- **Circular phase representation**: (cos φ, sin φ) instead of raw angles
- **RR-controlled thresholding**: Adaptive recurrence rate targeting
- **StratifiedGroupKFold**: Balanced cross-validation by subject
- **Retention-matched baseline**: Controls for selection bias
- **Dual HF band analysis**: Artifact validation for high-gamma

## RunPod Deployment

```bash
# On RunPod
git clone https://github.com/YOUR_USERNAME/eeg-state-biomarkers.git
cd eeg-state-biomarkers
uv sync

# Copy/mount your data
ln -s /path/to/your/data data

# Train
uv run python -m eeg_biomarkers.training.train data=full training.epochs=300
```

## License

MIT
