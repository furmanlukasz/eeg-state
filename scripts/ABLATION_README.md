# Ablation Study: Contrastive Loss Analysis

This ablation study addresses the reviewer concern that expert/novice effects
might be induced by the contrastive objective rather than genuine differences
in latent dynamics.

## Goal

Show that:
1. **NO_CONTRASTIVE**: The main group effect (latent speed differences) persists when contrastive loss is removed
2. **SHUFFLED_CONTRASTIVE**: Label-shuffled contrastive does NOT produce a consistent group effect

## Variants

| Variant | `lambda_contrastive` | Label shuffling | Description |
|---------|---------------------|-----------------|-------------|
| **full** | 0.1 | No | Baseline (current meditation model) |
| **no_contrastive** | 0.0 | N/A | Reconstruction only, no contrastive |
| **shuffled_contrastive** | 0.1 | Yes | Contrastive with random labels |

## What's Kept Fixed

- Preprocessing: 2-48 Hz bandpass, no re-referencing
- Chunking: 5s non-overlapping
- Model architecture: Transformer (hidden=64, 2 layers, 4 heads, ff=384)
- All other regularizers: `lambda_phase=1.0`, `lambda_amplitude=0.5`, `lambda_unit_circle=0.1`
- Subject splits: Same GroupKFold by subject, same seed
- Downstream metrics: Same PCA, same bootstrap settings

## Usage

### 1. Training on RunPod

```bash
# SSH into RunPod
ssh runpod

# Clone/pull repo
cd /workspace
git clone https://github.com/your-repo/eeg-state-biomarkers.git
cd eeg-state-biomarkers

# Install dependencies
pip install -e .

# Run all variants
python scripts/run_ablation_study.py \
    --data_dir /workspace/ds001787 \
    --output_dir outputs/ablation \
    --seed 42

# Or run specific variants
python scripts/run_ablation_study.py \
    --variants full no_contrastive \
    --data_dir /workspace/ds001787
```

### 2. Evaluation (can run locally after copying checkpoints)

```bash
# Copy checkpoints from RunPod
scp -r runpod:/workspace/eeg-state-biomarkers/outputs/ablation ./outputs/

# Run evaluation
python scripts/run_ablation_evaluation.py \
    --ablation_dir outputs/ablation \
    --data_dir /Volumes/Nvme_Data/ds001787 \
    --n_bootstrap 1000
```

### 3. Alternative: Run individual variants manually

```bash
# FULL (baseline)
python -m eeg_biomarkers.training.train \
    --config-name=experiment/ablation_full \
    paths.data_dir=/workspace/ds001787

# NO_CONTRASTIVE
python -m eeg_biomarkers.training.train \
    --config-name=experiment/ablation_no_contrastive \
    paths.data_dir=/workspace/ds001787

# SHUFFLED_CONTRASTIVE
python -m eeg_biomarkers.training.train \
    --config-name=experiment/ablation_shuffled_contrastive \
    paths.data_dir=/workspace/ds001787
```

## Output Structure

```
outputs/ablation/
├── full/
│   └── checkpoints/best.pt
├── no_contrastive/
│   └── checkpoints/best.pt
├── shuffled_contrastive/
│   └── checkpoints/best.pt
├── comparison/
│   ├── ablation_comparison_figure.png
│   ├── ablation_summary_table.csv
│   └── ablation_results.json
└── ablation_run_summary.json
```

## Key Outputs

### Comparison Figure
`ablation_comparison_figure.png` contains:
- Panel A: Speed difference (novice - expert) with 95% CI
- Panel B: Effect size (Cohen's d) with 95% CI
- Panel C: Reconstruction quality (val MSE)

### Summary Table
`ablation_summary_table.csv` contains per-variant:
- `epoch`: Best checkpoint epoch
- `val_loss`: Best validation loss
- `speed_difference`: Novice - expert mean speed
- `cohens_d`: Effect size
- Bootstrap confidence intervals

## Acceptance Criteria

1. **NO_CONTRASTIVE** should still show:
   - Main effect direction (novice > expert speed)
   - Non-trivial effect size (even if attenuated)

2. **SHUFFLED_CONTRASTIVE** should:
   - NOT systematically strengthen the group effect
   - Ideally weaken or add noise to the effect

3. **All variants** should have:
   - Comparable reconstruction quality (similar val_loss)
   - No catastrophic training failures

## Implementation Notes

### Label Shuffling
The shuffled contrastive variant creates a consistent random permutation of
group labels (controlled by the experiment seed). This permutation is used
ONLY for forming contrastive pairs during training. The true labels are
preserved for downstream group comparisons in evaluation.

See `src/eeg_biomarkers/training/enhanced_trainer.py:_get_shuffled_labels()`

### Config Files
- `configs/experiment/ablation_full.yaml`
- `configs/experiment/ablation_no_contrastive.yaml`
- `configs/experiment/ablation_shuffled_contrastive.yaml`

## Time Estimates

On RunPod with A100:
- Each variant: ~3-4 hours (600 epochs with early stopping at ~300)
- Total for 3 variants: ~10-12 hours

Consider running in parallel if resources allow:
```bash
# Run in separate tmux sessions
tmux new -s full
python scripts/run_ablation_study.py --variants full ...

tmux new -s no_contrastive
python scripts/run_ablation_study.py --variants no_contrastive ...
```
