# RunPod Training Commands

When the user asks to run training or ablation studies on RunPod GPU, provide commands in this format:

## Data Path Rules

- **RunPod**: `data/ds001787` (from workspace)
- **Local Mac**: `/Volumes/Nvme_Data/ds001787` (from NVMe volume)

## Standard Command Format

```bash
# Create logs directory if needed
mkdir -p logs

# Run training in background with nohup
nohup python -m eeg_biomarkers.training.train \
    --config-name=experiment/<CONFIG_NAME> \
    paths.data_dir=data/ds001787 \
    > logs/<CONFIG_NAME>.log 2>&1 &

# Monitor progress
tail -f logs/<CONFIG_NAME>.log
```

## Available Ablation Configs

### Broadband Ablations (original)
| Config Name | Description | Key Difference |
|-------------|-------------|----------------|
| `ablation_full` | Full model with contrastive | `lambda_contrastive: 0.1` |
| `ablation_no_contrastive` | No contrastive supervision | `lambda_contrastive: 0.0` |
| `ablation_shuffled_contrastive` | Shuffled labels | `contrastive_shuffle_labels: true` |
| `ablation_phase_only` | Phase only (no amplitude) | `include_amplitude: false` |

### Filterbank Ablations (narrowband - addresses "broadband Hilbert" critique)
| Config Name | Description | Key Difference |
|-------------|-------------|----------------|
| `ablation_filterbank_phase_only` | Narrowband phase only | 5 bands × 2 features (cos, sin) |
| `ablation_filterbank_full` | Narrowband phase + amplitude | 5 bands × 3 features |

**Bands**: delta (1-4), theta (4-8), alpha (8-13), beta (13-30), low_gamma (30-48)

## Example: Phase-Only Ablation on RunPod

```bash
mkdir -p logs

nohup python -m eeg_biomarkers.training.train \
    --config-name=experiment/ablation_phase_only \
    paths.data_dir=data/ds001787 \
    > logs/ablation_phase_only.log 2>&1 &

tail -f logs/ablation_phase_only.log
```

## Example: Run All Ablations Sequentially

```bash
mkdir -p logs

nohup bash -c '
    echo "=== Starting ablation_no_contrastive ===" && \
    python -m eeg_biomarkers.training.train \
        --config-name=experiment/ablation_no_contrastive \
        paths.data_dir=data/ds001787 && \
    echo "=== Starting ablation_phase_only ===" && \
    python -m eeg_biomarkers.training.train \
        --config-name=experiment/ablation_phase_only \
        paths.data_dir=data/ds001787 && \
    echo "=== All ablations complete ==="
' > logs/ablation_all.log 2>&1 &

tail -f logs/ablation_all.log
```

## Output Locations

Training outputs go to: `outputs/<experiment_name>/<dataset>/<timestamp>/`

For ablations:
- `outputs/ablation_no_contrastive/meditation_bids/<timestamp>/checkpoints/best.pt`
- `outputs/ablation_phase_only/meditation_bids/<timestamp>/checkpoints/best.pt`

## Example: Run Filterbank Ablations

```bash
mkdir -p logs

nohup bash -c '
    echo "=== Starting ablation_filterbank_phase_only ===" && \
    python -m eeg_biomarkers.training.train \
        --config-name=experiment/ablation_filterbank_phase_only \
        paths.data_dir=data/ds001787 && \
    echo "=== Starting ablation_filterbank_full ===" && \
    python -m eeg_biomarkers.training.train \
        --config-name=experiment/ablation_filterbank_full \
        paths.data_dir=data/ds001787 && \
    echo "=== Filterbank ablations complete ==="
' > logs/ablation_filterbank_all.log 2>&1 &

tail -f logs/ablation_filterbank_all.log
```

## Post-Training Analysis (Local Mac)

### Amplitude Ablation Analysis
After downloading checkpoints from RunPod, run amplitude ablation analysis:

```bash
EEG_DATASET=meditation_bids python scripts/local_analysis/amplitude_ablation_analysis.py \
    --phase-only-checkpoint outputs/ablation_phase_only/.../checkpoints/best.pt
```

### Filterbank Ablation Analysis
Compare narrowband vs broadband models:

```bash
EEG_DATASET=meditation_bids python scripts/local_analysis/filterbank_ablation_analysis.py \
    --filterbank-phase-only-checkpoint outputs/ablation_filterbank_phase_only/.../best.pt \
    --filterbank-full-checkpoint outputs/ablation_filterbank_full/.../best.pt \
    --broadband-phase-only-checkpoint outputs/ablation_phase_only/.../best.pt \
    --broadband-full-checkpoint outputs/ablation_no_contrastive/.../best.pt
```
