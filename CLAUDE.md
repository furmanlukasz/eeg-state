# MatrixAutoEncoder - Claude Code Context

## Project Overview

This repository implements a **ConvLSTM-based autoencoder framework** for analyzing resting-state EEG data to discover dynamical biomarkers of Mild Cognitive Impairment (MCI). The project is part of a PhD thesis investigating state-aware analysis of EEG signals.

### Central Research Question
> How can state-aware analysis of resting-state EEG reveal reliable dynamical biomarkers for Mild Cognitive Impairment (MCI)?

### Core Thesis
Resting EEG is a mixture of latent brain states; MCI alters the dynamics within and transitions between these states. Reliable biomarkers emerge only when features are computed within stable regimes, not across arbitrary continuous rest.

---

## Repository Structure

This repository has **two codebases**:

### 1. Modern Package: `src/eeg_biomarkers/` (PREFERRED)

A clean, modern Python package with proper structure:

```
src/eeg_biomarkers/
├── __init__.py
├── models/                    # Model architectures
│   ├── autoencoder.py        # ConvLSTMAutoencoder (main model)
│   ├── encoder.py            # ConvLSTMEncoder
│   └── decoder.py            # ConvLSTMDecoder
├── data/                      # Data loading & preprocessing
│   ├── dataset.py            # EEGDataset, EEGDataModule
│   └── preprocessing.py      # Phase extraction, chunking
├── training/                  # Training infrastructure
│   ├── trainer.py            # Training loop with early stopping
│   └── train.py              # Hydra entry point
├── analysis/                  # Analysis tools
│   ├── rqa.py                # RQA feature computation (Numba-optimized)
│   ├── state_discovery.py    # UMAP + clustering pipeline
│   └── classification.py     # XGBoost classification with GroupKFold
└── utils/                     # Utilities
    ├── device.py             # Device selection
    └── visualization.py      # Plotting functions
```

**Configuration**: `configs/` (Hydra YAML)
```
configs/
├── config.yaml               # Main config
├── model/                    # Model configs (base.yaml, complex.yaml)
├── data/                     # Data configs (default.yaml, full.yaml)
├── training/                 # Training configs
└── experiment/               # Experiment configs (integration_test.yaml)
```

### 2. Legacy Scripts: `scripts/` (REFERENCE ONLY)

The original research code. Useful for reference but has:
- Code duplication across classification scripts
- Manual configuration (no config system)
- Some mathematical issues (raw phase, fixed thresholds)

Use legacy scripts for:
- Understanding original implementation
- Accessing trained model checkpoints
- Running the Streamlit dashboard

---

## Quick Start (Modern Package)

```bash
# Install with UV
uv sync

# Or with pip
pip install -e ".[dev]"

# Train with default config
python -m eeg_biomarkers.training.train

# Train with custom config
python -m eeg_biomarkers.training.train model=complex training.epochs=200

# Run tests
pytest tests/
```

---

## Data Context

- **256-channel high-density resting-state EEG** (EGI GES 300 system, 250Hz)
- **78 MCI participants, 31 Healthy Controls**
- Eyes open/closed alternation during ~9 min recording
- Data stored as MNE `.fif` files in `data/{AD,HID,MCI}/` directories

---

## Critical Technical Notes

### Phase Representation (FIXED in modern package)
- **Problem** (legacy): L2 loss on wrapped angles [-π, π] is incorrect
- **Solution** (modern): Uses (cos φ, sin φ) circular representation
- See: `src/eeg_biomarkers/data/preprocessing.py:extract_phase_circular()`

### Label Leakage Prevention (FIXED in modern package)
- Uses `GroupKFold` by subject (not segment-level splits)
- UMAP/DBSCAN fitted on training data ONLY
- See: `src/eeg_biomarkers/data/dataset.py:EEGDataModule`
- See: `src/eeg_biomarkers/analysis/state_discovery.py:StateDiscovery`

### RQA Threshold Selection (FIXED in modern package)
- Uses RR-controlled thresholding (target 1-5% recurrence rate)
- Configurable via `configs/model/*.yaml`
- See: `src/eeg_biomarkers/models/autoencoder.py:compute_recurrence_matrix()`

### High-Gamma Artifact Concerns
- EMG contamination risk above 40Hz (especially temporal/frontal electrodes)
- Central/occipital electrodes more reliable for gamma analysis
- Require ICA inspection and topographic validation

---

## Key Commands

### Modern Package
```bash
# Training
python -m eeg_biomarkers.training.train

# With config overrides
python -m eeg_biomarkers.training.train model=complex data=full training.epochs=200

# Multi-run sweep
python -m eeg_biomarkers.training.train --multirun model=base,complex

# Tests
pytest tests/ -v
pytest tests/test_models.py -v  # Specific test file
```

### Legacy (Makefile)
```bash
make train_model              # Train autoencoder
make generate_rm              # Generate recurrence matrices
make visualize_latent_space   # UMAP visualizations
make run_streamlit            # Launch Streamlit app
```

---

## Dependencies

Managed via `pyproject.toml` with UV:

**Core:**
- `torch>=2.0`: Deep learning
- `mne>=1.5`: EEG processing
- `hydra-core>=1.3`: Configuration

**Analysis:**
- `umap-learn>=0.5`: Dimensionality reduction
- `xgboost>=2.0`: Classification
- `numba>=0.58`: RQA optimization

**Development:**
- `pytest>=7.0`: Testing
- `ruff>=0.1`, `black>=23.0`: Linting/formatting

Install with:
```bash
uv sync                    # All dependencies
uv sync --extra dev        # Include dev tools
```

---

## Testing

Tests are in `tests/`:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=eeg_biomarkers --cov-report=term-missing

# Run specific tests
pytest tests/test_preprocessing.py -v
pytest tests/test_models.py::TestConvLSTMAutoencoder -v
```

Key test files:
- `test_preprocessing.py`: Phase extraction, chunking
- `test_models.py`: Encoder, decoder, autoencoder
- `test_rqa.py`: RQA feature computation

---

## Configuration (Hydra)

Override any config via command line:

```bash
# Change model
python -m eeg_biomarkers.training.train model=complex

# Change multiple settings
python -m eeg_biomarkers.training.train \
    model.encoder.hidden_size=128 \
    training.epochs=50 \
    experiment.name="my_experiment"

# Run integration experiment
python -m eeg_biomarkers.training.train \
    --config-name=experiment/integration_test
```

---

## Current Research Focus

1. **Integration Experiment**: Validate that state-conditioning improves subject-level generalization
2. **Artifact Validation**: Confirm high-gamma findings are neural, not EMG
3. **Robustness Analysis**: Test across RR thresholds (1%, 2%, 5%)

---

## Brain Criticality Framework

The theoretical interpretation:
- MCI represents shift toward **supercriticality** (not sub-criticality)
- Increased determinism → attractor collapse
- Higher LAM/DET → more rigid dynamics
- Reduced entropy → simplified state space

RQA findings support this: increased recurrence/determinism in MCI indicates departure from optimal "edge of chaos" dynamics.

---

## References to Context Bus

Project slug: `phd-eeg-mci-biomarkers`

Key artifacts available:
- PhD thesis summary and plan
- Technical specifications for autoencoder fixes
- Literature review (2022-2025)
- Priority integration experiment design

---

## Migration Guide: Legacy → Modern

| Legacy Location | Modern Location |
|-----------------|-----------------|
| `scripts/utils.py:ConvLSTMEEGAutoencoder` | `src/eeg_biomarkers/models/autoencoder.py` |
| `scripts/train_model.py` | `src/eeg_biomarkers/training/train.py` |
| `scripts/utils.py:extract_phase()` | `src/eeg_biomarkers/data/preprocessing.py:extract_phase_circular()` |
| `scripts/cloud-pod/RQA.py` | `src/eeg_biomarkers/analysis/rqa.py` |
| `scripts/app/classification_cv.py` | `src/eeg_biomarkers/analysis/classification.py` |

Key improvements in modern package:
- Circular phase representation (cos, sin)
- RR-controlled thresholding
- Proper GroupKFold by subject
- Hydra configuration
- Type hints throughout
- pytest test suite

---

## Current Session Status (January 2026)

### What We Did

1. **Fixed torch.acos segfault** - Tighter clamping for numerical stability
2. **Added dual HF band analysis** - HF1 (30-48Hz) + HF2 (70-110Hz) for artifact control
3. **Created integration experiment** with 12 guardrails:
   - Retention-matched baseline (controls for selection bias)
   - Class-conditional retention audit (HC vs MCI per fold)
   - Per-fold subject distribution logging
   - Bootstrap CIs per RR target
4. **Trained new model** with modern architecture (val_loss=0.3128 on subset)
5. **Fixed critical AUC computation bug** in `classification.py`:
   - Was using hard labels (0/1) for subject-level AUC
   - Now uses mean probabilities per subject (correct ROC-AUC)
6. **Implemented StratifiedGroupKFold** for balanced HC/MCI per fold
7. **Added single-class fold handling** - skips folds with only one class

### What Failed / Key Issues Found

1. **Null model = real model AUC** - Critic agent identified that state-conditioning improvement was NOT from meaningful state structure, but from selection bias (which segments survive HDBSCAN clustering)
2. **Subject-level AUC was computed incorrectly** - Using hard predictions instead of probabilities made AUC extremely unstable with 3-4 test subjects per fold
3. **Unbalanced folds** - GroupKFold created folds like (1 HC, 3 MCI) making AUC nearly meaningless
4. **Low subject count** - Only 9 HC / 10 MCI in test subset = underpowered for reliable AUC

### Next Steps (RunPod GPU)

1. **Push code to GitHub** with all fixes
2. **Set up RunPod environment**:
   ```bash
   git clone <repo>
   cd MatrixAutoEncoder
   uv sync
   ```
3. **Train on FULL dataset** (78 MCI + 31 HC):
   ```bash
   uv run python -m eeg_biomarkers.training.train \
       data=full \
       training.epochs=300 \
       model.encoder.hidden_size=128
   ```
4. **Run integration experiment** with all controls:
   ```bash
   uv run python -m eeg_biomarkers.experiments.integration_experiment \
       --data-dir data \
       --output-dir results/integration_v3 \
       --checkpoint models/best.pt \
       --n-folds 5 \
       --n-seeds 3
   ```
5. **Critical analysis**: Check if `state_conditioned AUC > retention_matched_baseline AUC`
   - If YES → state structure provides real benefit
   - If NO → improvement is just selection bias

### Key Files Modified

| File | Change |
|------|--------|
| `src/eeg_biomarkers/analysis/classification.py` | AUC from probabilities, not hard labels |
| `src/eeg_biomarkers/experiments/integration_experiment.py` | StratifiedGroupKFold, single-class handling, retention-matched baseline |
| `tests/test_integration_experiment.py` | Fixed to use 3D trajectory arrays |

### Tests Status

All 15 tests pass with `uv run python -m pytest tests/`
