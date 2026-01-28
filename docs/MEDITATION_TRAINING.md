# Meditation Dataset Training Guide

Training guide for the OpenNeuro Meditation/Mind-wandering dataset (ds001787).

**Dataset**: Expert vs Novice Meditators
**Source**: https://openneuro.org/datasets/ds001787
**Subjects**: 12 experts + 12 novices = 24 total
**Channels**: 64 EEG (BioSemi BDF format)

---

## Dataset Setup

### 1. Download from OpenNeuro

```bash
# Using datalad (recommended for large datasets)
datalad install https://github.com/OpenNeuroDatasets/ds001787.git
cd ds001787
datalad get .  # Download all files

# Or using openneuro-py
pip install openneuro-py
openneuro download --dataset ds001787 --target-dir /workspace/data/ds001787
```

### 2. Verify Structure

```bash
ls /workspace/data/ds001787/
# Should see:
# participants.tsv  sub-001/  sub-002/  ...  sub-024/

cat /workspace/data/ds001787/participants.tsv
# participant_id  group    age  ...
# sub-001         expert   45   ...
# sub-013         novice   32   ...
```

---

## Two-Phase Training Workflow

Same workflow as Greek resting-state, but with meditation data.

### Phase 1: Reconstruction Training

```bash
nohup python -m eeg_biomarkers.training.train \
    data=meditation_optimized \
    model=transformer_v2 \
    training=optimized \
    paths.data_dir=/workspace/data/ds001787 \
    > training_meditation_phase1.log 2>&1 &
```

**Monitor:**
```bash
tail -f training_meditation_phase1.log
```

**Phase 1 completion criteria:**
- `val_loss < 0.5` (good reconstruction)
- Loss plateau or early stopping triggered
- Typically 100-200 epochs

### Verification

Before Phase 2, verify reconstruction quality:

```bash
# Find best checkpoint
ls outputs/*/checkpoints/

# Test reconstruction
python scripts/generate_eeg.py \
    --checkpoint outputs/<run_name>/checkpoints/best.pt \
    --mode reconstruct \
    --input-file /workspace/data/ds001787/sub-001/ses-01/eeg/sub-001_ses-01_task-meditation_eeg.bdf \
    --output meditation_reconstruction_check.png
```

**What to check:**
- Reconstructed signal matches original phase structure
- No artifacts or noise amplification
- Smooth latent space (try `--mode interpolate`)

### Phase 2: Contrastive Training

```bash
nohup python -m eeg_biomarkers.training.train \
    data=meditation_optimized \
    model=transformer_v2 \
    training=optimized_phase2 \
    paths.data_dir=/workspace/data/ds001787 \
    training.resume_from="outputs/<run_name>/checkpoints/best.pt" \
    > training_meditation_phase2.log 2>&1 &
```

**Monitor:**
```bash
tail -f training_meditation_phase2.log
```

**Phase 2 completion criteria:**
- Contrastive loss (`contr`) decreasing
- Total loss stable
- Early stopping or 150 epochs

---

## Quick Training Commands

### Basic Training (Single Command)

```bash
nohup python -m eeg_biomarkers.training.train \
    data=meditation_optimized \
    model=transformer_v2 \
    training=optimized \
    paths.data_dir=/workspace/data/ds001787 \
    > training_meditation.log 2>&1 &
```

### Memory-Efficient Mode

If RAM is limited (dataset is ~24 subjects, smaller than Greek):

```bash
nohup python -m eeg_biomarkers.training.train \
    data=meditation_optimized \
    model=transformer_v2 \
    training=optimized \
    paths.data_dir=/workspace/data/ds001787 \
    +data.sampling.preload_to_ram=false \
    > training_meditation.log 2>&1 &
```

### With Average Reference (Original Paper)

Original paper used average reference instead of CSD:

```bash
nohup python -m eeg_biomarkers.training.train \
    data=meditation_optimized \
    model=transformer_v2 \
    training=optimized \
    paths.data_dir=/workspace/data/ds001787 \
    data.preprocessing.reference=average \
    > training_meditation.log 2>&1 &
```

---

## Dataset Comparison

| Property | Greek Resting | Meditation BIDS |
|----------|--------------|-----------------|
| Subjects | 78 MCI + 31 HC + AD | 12 expert + 12 novice |
| Channels | 256 (EGI) | 64 (BioSemi) |
| Format | .fif | .bdf |
| Paradigm | Eyes open/closed | Meditation with probes |
| Window | 10s | 10s |
| Groups | Disease status | Expertise level |

**Key differences:**
- Meditation has fewer subjects (24 vs 109+) but more sessions per subject
- 64 vs 256 channels = smaller model input dimension
- BDF format requires different preprocessing path
- Group structure from `participants.tsv` not folder names

---

## Configuration Reference

| Config | Description |
|--------|-------------|
| `data=meditation_optimized` | Expert vs Novice, 10s windows, CSD reference |
| `data=meditation_bids` | Original config, 5s windows, average reference |

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `paths.data_dir` | Required | Path to ds001787 root |
| `data.preprocessing.reference` | csd | csd or average |
| `data.preprocessing.chunk_duration` | 10.0 | Window size in seconds |
| `data.preprocessing.chunk_overlap` | 5.0 | Overlap in seconds |

---

## Troubleshooting

### "Participants file not found"

Make sure the dataset is fully downloaded:
```bash
ls /workspace/data/ds001787/participants.tsv
```

### "Git-annex file not downloaded"

The BDF files are large and may be git-annex symlinks:
```bash
cd /workspace/data/ds001787
datalad get sub-*/ses-*/eeg/*.bdf
```

### Channel Mismatch Errors

The model expects 64 EEG channels. Verify:
```python
import mne
raw = mne.io.read_raw_bdf("/workspace/data/ds001787/sub-001/ses-01/eeg/sub-001_ses-01_task-meditation_eeg.bdf")
print(len(mne.pick_types(raw.info, eeg=True)))  # Should be 64
```

### CSD Reference Fails

CSD requires channel positions. If missing, fall back to average:
```bash
data.preprocessing.reference=average
```

---

## Complete Example

```bash
# === SETUP ===
# Download dataset (do once)
openneuro download --dataset ds001787 --target-dir /workspace/data/ds001787

# === PHASE 1: Reconstruction ===
nohup python -m eeg_biomarkers.training.train \
    data=meditation_optimized \
    model=transformer_v2 \
    training=optimized \
    paths.data_dir=/workspace/data/ds001787 \
    > training_meditation_phase1.log 2>&1 &

# Wait for completion
tail -f training_meditation_phase1.log

# === VERIFICATION ===
ls outputs/*/checkpoints/
python scripts/generate_eeg.py \
    --checkpoint outputs/<run>/checkpoints/best.pt \
    --mode reconstruct \
    --input-file /workspace/data/ds001787/sub-001/ses-01/eeg/sub-001_ses-01_task-meditation_eeg.bdf \
    --output verify_meditation.png

# === PHASE 2: Contrastive ===
nohup python -m eeg_biomarkers.training.train \
    data=meditation_optimized \
    model=transformer_v2 \
    training=optimized_phase2 \
    paths.data_dir=/workspace/data/ds001787 \
    training.resume_from="outputs/<run>/checkpoints/best.pt" \
    > training_meditation_phase2.log 2>&1 &
```

---

## WandB

Training logs to Weights & Biases. View at:
https://wandb.ai/lfurman_108/eeg-mci-biomarkers
