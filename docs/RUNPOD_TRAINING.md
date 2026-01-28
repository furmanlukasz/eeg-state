# RunPod Training Guide

Quick reference for training EEG biomarker models on RunPod GPU instances.

## Setup

```bash
# Clone repository
git clone https://github.com/furmanlukasz/eeg-state.git
cd eeg-state

# Install dependencies
pip install -e .

# Fix DNS if needed (common RunPod issue)
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
```

---

## Two-Phase Training Workflow

Training uses a curriculum approach:
1. **Phase 1**: Reconstruction only (learn good representations)
2. **Verification**: Check reconstruction quality
3. **Phase 2**: Add contrastive loss (learn condition-discriminative features)

### Phase 1: Reconstruction Training

Train the model to reconstruct EEG phase representations. No contrastive loss.

```bash
nohup python -m eeg_biomarkers.training.train \
    data=optimized_all \
    model=transformer_v2 \
    training=optimized \
    > training_phase1.log 2>&1 &
```

**Monitor progress:**
```bash
tail -f training_phase1.log
```

**Phase 1 completion criteria:**
- `val_loss < 0.5` (good reconstruction)
- Loss has plateaued (early stopping triggered or epochs complete)
- Training typically takes 100-300 epochs

### Phase 1 → Phase 2: Verification

Before starting Phase 2, verify the model learned good representations.

**1. Find your best checkpoint:**
```bash
ls -la outputs/*/checkpoints/
# Look for: best.pt or checkpoint with lowest val_loss
```

**2. Test reconstruction quality:**
```bash
python scripts/generate_eeg.py \
    --checkpoint outputs/<run_name>/checkpoints/best.pt \
    --mode reconstruct \
    --input-file data/HID/<any_subject>.fif \
    --output reconstruction_check.png
```

**What to look for:**
- Reconstructed signal should visually match original
- Phase structure should be preserved
- No obvious artifacts or noise amplification

**3. Test latent space smoothness (optional):**
```bash
python scripts/generate_eeg.py \
    --checkpoint outputs/<run_name>/checkpoints/best.pt \
    --mode interpolate \
    --input-file data/HID/<any_subject>.fif \
    --output interpolation_check.png \
    --n-steps 10
```

**What to look for:**
- Smooth transitions between samples
- No sudden jumps or artifacts
- Indicates well-structured latent space

### Phase 2: Contrastive Training

After verification passes, add contrastive loss to learn condition-discriminative features.

```bash
nohup python -m eeg_biomarkers.training.train \
    data=optimized_all \
    model=transformer_v2 \
    training=optimized_phase2 \
    training.resume_from="outputs/<run_name>/checkpoints/best.pt" \
    > training_phase2.log 2>&1 &
```

**Monitor progress:**
```bash
tail -f training_phase2.log
```

**Phase 2 key differences:**
- Lower learning rate (0.0001 vs 0.0003)
- Contrastive loss weight: 0.15
- Reconstruction weight: 0.8 (slightly reduced)
- Shorter training: 150 epochs max

**Phase 2 completion criteria:**
- Contrastive loss (`contr`) is decreasing
- Total loss has stabilized
- Early stopping triggered or 150 epochs complete

---

## Quick Training Commands

### Basic Training (MCI vs HC)

```bash
nohup python -m eeg_biomarkers.training.train \
    data=optimized \
    model=transformer_v2 \
    training=optimized \
    > training.log 2>&1 &
```

### All Groups (AD + MCI + HC)

```bash
nohup python -m eeg_biomarkers.training.train \
    data=optimized_all \
    model=transformer_v2 \
    training=optimized \
    > training.log 2>&1 &
```

### AD vs HC Only

```bash
nohup python -m eeg_biomarkers.training.train \
    data=optimized_ad \
    model=transformer_v2 \
    training=optimized \
    > training.log 2>&1 &
```

### Memory-Efficient Mode (Low RAM)

If running out of RAM (~40-50GB needed for full preload), use LRU cache mode:

```bash
nohup python -m eeg_biomarkers.training.train \
    data=optimized_all \
    model=transformer_v2 \
    training=optimized \
    +data.sampling.preload_to_ram=false \
    > training.log 2>&1 &
```

### Disable Caching (Disk Space Issues)

```bash
nohup python -m eeg_biomarkers.training.train \
    data=optimized_all \
    model=transformer_v2 \
    training=optimized \
    data.caching.enabled=false \
    > training.log 2>&1 &
```

### Larger Batch Size (More VRAM Available)

If GPU VRAM usage is low (<50%), increase batch size:

```bash
nohup python -m eeg_biomarkers.training.train \
    data=optimized_all \
    model=transformer_v2 \
    training=optimized \
    training.batch_size=64 \
    > training.log 2>&1 &
```

---

## Monitoring

### Watch Training Progress

```bash
tail -f training.log
```

### Check if Training is Running

```bash
ps aux | grep python
```

### Check GPU Usage

```bash
watch -n 1 nvidia-smi
```

### Check Disk Space

```bash
df -h /workspace
du -sh /workspace/*
```

### Understanding Training Logs

Example log line:
```
Epoch 50: loss=0.4123 ph=0.35 amp=0.04 uc=0.01 ang=0.01 dv=0.003 contr=0.0
```

| Field | Meaning |
|-------|---------|
| `loss` | Total weighted loss |
| `ph` | Phase reconstruction loss (MSE on cos/sin) |
| `amp` | Amplitude reconstruction loss |
| `uc` | Unit circle regularization |
| `ang` | Angle consistency loss |
| `dv` | Derivative loss (temporal smoothness) |
| `contr` | Contrastive loss (0 in Phase 1, >0 in Phase 2) |

---

## Troubleshooting

### DNS Resolution Failed

```bash
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
git pull
```

### Out of RAM

Add `+data.sampling.preload_to_ram=false` to use LRU cache instead of preloading all data.

### Out of Disk Space

```bash
# Remove old caches
rm -rf preprocessed_data*/
rm -rf outputs/
rm -rf wandb/

# Or disable caching
data.caching.enabled=false
```

### Process Killed / OOM

Reduce batch size:
```bash
training.batch_size=16
```

### "Key not in struct" Error

Use `+` prefix for new config keys:
```bash
+data.sampling.preload_to_ram=false  # Correct
data.sampling.preload_to_ram=false   # Error
```

---

## Configuration Reference

| Config | Description |
|--------|-------------|
| `data=optimized` | MCI vs HC, 10s windows, 50% overlap |
| `data=optimized_ad` | AD vs HC |
| `data=optimized_all` | AD + MCI + HC (3-class) |
| `model=transformer_v2` | Large transformer (320 hidden, 6 layers) |
| `training=optimized` | Phase 1: 300 epochs, reconstruction only |
| `training=optimized_phase2` | Phase 2: 150 epochs, contrastive enabled |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.epochs` | 300 (P1) / 150 (P2) | Max training epochs |
| `training.batch_size` | 32 (P1) / 64 (P2) | Batch size |
| `training.early_stopping.patience` | 40 (P1) / 25 (P2) | Early stopping patience |
| `training.lambda_contrastive` | 0.0 (P1) / 0.15 (P2) | Contrastive loss weight |
| `training.optimizer.lr` | 0.0003 (P1) / 0.0001 (P2) | Learning rate |
| `+data.sampling.preload_to_ram` | true | Load all data to RAM (~40GB) |
| `data.caching.enabled` | true | Cache preprocessed data to disk |

---

## WandB

Training logs to Weights & Biases automatically. View at:
https://wandb.ai/lfurman_108/eeg-mci-biomarkers

---

## Complete Two-Phase Example

```bash
# === PHASE 1: Reconstruction ===
nohup python -m eeg_biomarkers.training.train \
    data=optimized_all \
    model=transformer_v2 \
    training=optimized \
    > training_phase1.log 2>&1 &

# Wait for completion, then verify
tail -f training_phase1.log

# === VERIFICATION ===
# Find best checkpoint
ls outputs/*/checkpoints/

# Test reconstruction
python scripts/generate_eeg.py \
    --checkpoint outputs/<run>/checkpoints/best.pt \
    --mode reconstruct \
    --input-file data/HID/sub-001_eeg.fif \
    --output verify_reconstruction.png

# === PHASE 2: Contrastive ===
nohup python -m eeg_biomarkers.training.train \
    data=optimized_all \
    model=transformer_v2 \
    training=optimized_phase2 \
    +training.resume_from="outputs/<run>/checkpoints/best.pt" \
    > training_phase2.log 2>&1 &
```
