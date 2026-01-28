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

## Training Commands

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

## Configuration Reference

| Config | Description |
|--------|-------------|
| `data=optimized` | MCI vs HC, 10s windows, 50% overlap |
| `data=optimized_ad` | AD vs HC |
| `data=optimized_all` | AD + MCI + HC (3-class) |
| `model=transformer_v2` | Large transformer (320 hidden, 6 layers) |
| `training=optimized` | 300 epochs, patience 40, geometric loss |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.epochs` | 300 | Max training epochs |
| `training.batch_size` | 32 | Batch size (effective 64 with accumulation) |
| `training.early_stopping.patience` | 40 | Early stopping patience |
| `+data.sampling.preload_to_ram` | true | Load all data to RAM (fast but ~40GB) |
| `data.caching.enabled` | true | Cache preprocessed data to disk |

## WandB

Training logs to Weights & Biases automatically. View at:
https://wandb.ai/lfurman_108/eeg-mci-biomarkers
