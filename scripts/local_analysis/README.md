# Local Analysis Scripts

Scripts for analyzing EEG autoencoder results on your local M1 MacBook.

## Setup

1. **Edit `config.py`** to point to your local paths:
   ```python
   CHECKPOINT_PATH = Path("/Users/luki/Documents/GitHub/MatrixAutoEncoder/models/best.pt")
   DATA_DIR = Path("/Users/luki/Documents/GitHub/MatrixAutoEncoder/data")
   ```

2. **Install dependencies** (if not already installed):
   ```bash
   cd /Users/luki/Documents/GitHub/eeg-state-biomarkers
   uv sync
   ```

3. **Activate environment**:
   ```bash
   source .venv/bin/activate
   ```

## Available Scripts

### 1. Plot Recurrence Matrices
```bash
# Default: 2 subjects, chunk 0, Theiler=50
python scripts/local_analysis/plot_recurrence.py

# More subjects
python scripts/local_analysis/plot_recurrence.py --n-subjects 5

# Filter by condition (HID=healthy, MCI, AD)
python scripts/local_analysis/plot_recurrence.py --conditions MCI
python scripts/local_analysis/plot_recurrence.py --conditions HID MCI

# Specific subject
python scripts/local_analysis/plot_recurrence.py --subject S001

# List available subjects first
python scripts/local_analysis/plot_recurrence.py --list-subjects

# List chunks for a subject
python scripts/local_analysis/plot_recurrence.py --list-chunks S001

# Select specific chunk (0-indexed)
python scripts/local_analysis/plot_recurrence.py --chunk 3

# Plot ALL chunks for a subject
python scripts/local_analysis/plot_recurrence.py --subject S001 --chunk all

# Disable Theiler window (shows diagonal structure)
python scripts/local_analysis/plot_recurrence.py --no-theiler

# Custom Theiler window (samples)
python scripts/local_analysis/plot_recurrence.py --theiler 100

# Don't show interactive plots (just save)
python scripts/local_analysis/plot_recurrence.py --no-show
```

### 2. 3D UMAP Visualization
```bash
# Default: HC vs MCI comparison (no AD)
python scripts/local_analysis/plot_umap_3d.py

# Compare specific conditions
python scripts/local_analysis/plot_umap_3d.py --conditions HID MCI      # HC vs MCI only
python scripts/local_analysis/plot_umap_3d.py --conditions HID AD       # HC vs AD only
python scripts/local_analysis/plot_umap_3d.py --conditions HID MCI AD   # All three groups

# More subjects per group
python scripts/local_analysis/plot_umap_3d.py --n-subjects 5

# More chunks per subject
python scripts/local_analysis/plot_umap_3d.py --n-chunks 10

# Only mean latent UMAP
python scripts/local_analysis/plot_umap_3d.py --mode mean

# Only trajectory UMAP (all timepoints - slow!)
python scripts/local_analysis/plot_umap_3d.py --mode trajectory

# Only per-subject coloring
python scripts/local_analysis/plot_umap_3d.py --mode subject

# List available subjects
python scripts/local_analysis/plot_umap_3d.py --list-subjects
```

### 3. Compare Groups (HC vs MCI vs AD)
```bash
# Default: HC vs MCI comparison (3 subjects per group)
python scripts/local_analysis/compare_hc_mci.py

# Compare specific conditions
python scripts/local_analysis/compare_hc_mci.py --conditions HID MCI      # HC vs MCI only
python scripts/local_analysis/compare_hc_mci.py --conditions HID AD       # HC vs AD only
python scripts/local_analysis/compare_hc_mci.py --conditions HID MCI AD   # All three groups

# More subjects
python scripts/local_analysis/compare_hc_mci.py --n-subjects 5

# Average RQA over multiple chunks
python scripts/local_analysis/compare_hc_mci.py --n-chunks 3

# Different RR target
python scripts/local_analysis/compare_hc_mci.py --rr-target 0.05

# List available subjects
python scripts/local_analysis/compare_hc_mci.py --list-subjects
```

**Outputs:**
- **Recurrence matrices** side-by-side for each group
- **Violin plots** with all RQA features (DET, LAM, TT, L_mean, ENTR, DIV)
- **Expected trends** note printed to console (based on brain criticality framework)

**RQA Features:**
| Feature | Description | Expected in MCI |
|---------|-------------|-----------------|
| DET | Determinism (diagonal lines) | ↑ higher |
| LAM | Laminarity (vertical lines) | ↑ higher |
| TT | Trapping Time | ↑ higher |
| L_mean | Mean diagonal length | ↑ higher |
| ENTR | Entropy of diagonals | ↓ lower |
| DIV | Divergence (1/L_max) | ↓ lower |

### 4. RQA-based Classification
```bash
# Default: HC vs MCI, 10 subjects per group, 3 chunks each
python scripts/local_analysis/classify_rqa.py

# Compare specific conditions
python scripts/local_analysis/classify_rqa.py --conditions HID MCI      # HC vs MCI
python scripts/local_analysis/classify_rqa.py --conditions HID AD       # HC vs AD
python scripts/local_analysis/classify_rqa.py --conditions HID MCI AD   # HC vs all impaired

# More subjects/chunks
python scripts/local_analysis/classify_rqa.py --n-subjects 15
python scripts/local_analysis/classify_rqa.py --n-chunks 5

# Different RR target
python scripts/local_analysis/classify_rqa.py --rr-target 0.05

# Different test split
python scripts/local_analysis/classify_rqa.py --test-size 0.3

# Custom Theiler window
python scripts/local_analysis/classify_rqa.py --theiler 100
python scripts/local_analysis/classify_rqa.py --no-theiler

# BASELINE: Use random weights (untrained model)
python scripts/local_analysis/classify_rqa.py --random-weights

# List available subjects
python scripts/local_analysis/classify_rqa.py --list-subjects
```

**Outputs:**
- **ROC curve** with AUC score (`roc_curve.png`)
- **Feature importance** plot (`feature_importance.png`)
- **Confusion matrix** (`confusion_matrix.png`)
- Console output with segment-level and subject-level metrics

**Classification approach:**
- Binary classification: HC (label=0) vs Impaired (MCI/AD, label=1)
- Train/test split by subject (no leakage)
- Uses XGBoost (or RandomForest if XGBoost not available)
- Reports both segment-level and subject-level AUC
- `--random-weights` flag provides baseline: if trained model AUC >> random AUC, the learned representations are meaningful

### 5. Rabinovich-style Trajectory Visualization
```bash
# Default: first HC subject, all visualizations
python scripts/local_analysis/plot_trajectory.py

# Specific subject
python scripts/local_analysis/plot_trajectory.py --subject S001

# More chunks for longer trajectory
python scripts/local_analysis/plot_trajectory.py --n-chunks 10

# Specific visualization mode
python scripts/local_analysis/plot_trajectory.py --mode trajectory   # 2D/3D PCA
python scripts/local_analysis/plot_trajectory.py --mode speed        # Speed + metastability
python scripts/local_analysis/plot_trajectory.py --mode flow         # Flow field (quiver)
python scripts/local_analysis/plot_trajectory.py --mode density      # Density heatmap

# Compare trained vs random weights
python scripts/local_analysis/plot_trajectory.py --compare-random

# Use MCI subject
python scripts/local_analysis/plot_trajectory.py --conditions MCI

# List available subjects
python scripts/local_analysis/plot_trajectory.py --list-subjects
```

**Outputs:**
- **2D/3D trajectory** colored by time (`trajectory_2d.png`, `trajectory_3d.png`)
- **Speed-colored trajectory** with metastable region markers (`trajectory_speed.png`)
- **Density heatmap** showing where system spends time (`trajectory_density.png`)
- **Flow field (quiver)** showing local displacement vectors (`trajectory_flow.png`)
- **Trained vs random comparison** if `--compare-random` flag used

**Trajectory statistics computed:**
| Statistic | Description |
|-----------|-------------|
| Mean speed | Average velocity in latent space |
| Speed CV | Coefficient of variation (consistency) |
| Dwell episodes | Number of slow/metastable periods |
| Tortuosity | Path complexity (path length / displacement) |
| Explored variance | Volume of latent space explored |

**Theoretical basis (Rabinovich IFPS):**
- Trajectories reveal "channels" and preferred flow directions
- Metastable regions (slow speed) indicate attractor-like states
- MCI may show: fewer channels, longer dwell times, reduced flexibility
- If trained >> random, learned representations capture meaningful dynamics

### 6. Multi-Embedding Trajectory Analysis
```bash
# Default: HC vs MCI comparison with all embedding methods
python scripts/local_analysis/multi_embedding.py

# Specific subject only (no group comparison)
python scripts/local_analysis/multi_embedding.py --subject S001

# More subjects per group
python scripts/local_analysis/multi_embedding.py --n-subjects 5

# More chunks for longer trajectories
python scripts/local_analysis/multi_embedding.py --n-chunks 10

# Custom time lag for tPCA and delay embedding
python scripts/local_analysis/multi_embedding.py --tau 10

# Custom delay embedding dimension
python scripts/local_analysis/multi_embedding.py --delay-dim 4

# Compare trained vs random weights
python scripts/local_analysis/multi_embedding.py --compare-random

# Use MCI subjects
python scripts/local_analysis/multi_embedding.py --conditions MCI

# List available subjects
python scripts/local_analysis/multi_embedding.py --list-subjects
```

**Outputs:**
- **Multi-panel embedding comparison** showing all 5 methods side-by-side (`multi_embedding_comparison.png`)
- **Flow metrics bar chart** comparing methods (`flow_metrics_comparison.png`)
- **Cross-embedding consistency matrix** with Spearman correlations (`embedding_consistency.png`)
- **Group comparison** (HC vs MCI) for each embedding method (`group_comparison.png`)
- **Trained vs random comparison** if `--compare-random` flag used

**Embedding methods:**
| Method | Description | Key Parameters |
|--------|-------------|----------------|
| PCA | Standard linear projection | n_components=2 |
| Time-lagged PCA | Captures temporal structure via lagged covariance | tau (lag samples) |
| Diffusion Maps | Nonlinear, respects manifold geometry | k (neighbors), epsilon |
| UMAP | Preserves local + global structure | n_neighbors, min_dist |
| Delay Embedding | Takens reconstruction of dynamics | tau, dim |

**Flow geometry metrics computed:**
| Metric | Description | Expected in MCI |
|--------|-------------|-----------------|
| Mean speed | Average velocity in embedding | ↓ lower (more rigid) |
| Speed CV | Coefficient of variation | ↓ lower (less variable) |
| Dwell episodes | # of slow/metastable periods | ↑ higher |
| Occupancy entropy | Uniformity of space exploration | ↓ lower |
| Tortuosity | Path complexity (path/displacement) | ↓ lower |
| Explored variance | Volume of space visited | ↓ lower |

**Cross-embedding consistency:**
- Computes pairwise Spearman correlation of flow metrics across methods
- High consistency (>0.7) suggests robust geometric features
- Low consistency may indicate method-specific artifacts
- Theory: if embeddings agree, the underlying dynamics are captured reliably

**Theoretical basis:**
- Multiple embeddings provide complementary views of the same dynamics
- Time-lagged PCA captures temporal correlations missed by standard PCA
- Diffusion maps respect the intrinsic manifold geometry
- Delay embedding (Takens) reconstructs attractor structure
- Cross-method agreement validates that findings are not embedding artifacts

### 7. Full-Dataset Statistical Analysis (Publication-Ready)
```bash
# Fast analysis (pca, tpca, delay) - default, recommended
python scripts/local_analysis/full_dataset_analysis.py

# Run without showing plots (saves automatically)
python scripts/local_analysis/full_dataset_analysis.py --no-show

# Quick test mode (100 bootstrap, 5 subjects per group)
python scripts/local_analysis/full_dataset_analysis.py --quick --no-show

# Include slow methods (diffusion maps, UMAP) - takes much longer
python scripts/local_analysis/full_dataset_analysis.py --embedding all

# Single embedding method
python scripts/local_analysis/full_dataset_analysis.py --embedding pca

# Custom bootstrap iterations
python scripts/local_analysis/full_dataset_analysis.py --n-bootstrap 1000

# More chunks per subject for longer trajectories
python scripts/local_analysis/full_dataset_analysis.py --n-chunks 15
```

**Embedding options:**
| Option | Methods | Speed |
|--------|---------|-------|
| `--embedding fast` (default) | PCA, tPCA, Delay | Fast |
| `--embedding all` | PCA, tPCA, Delay, Diffusion, UMAP | Slow |
| `--embedding pca` | PCA only | Fastest |

**Purpose:**
This script provides publication-ready statistical analysis for a systems neuroscience paper. It focuses on **quantifying robust reorganization of metastable brain dynamics** across groups, NOT classification or biomarkers.

**Key features:**
1. **Subject-level bootstrapping** (500+ iterations) for confidence intervals
2. **Density difference maps with statistical masking** (95% CI excludes 0)
3. **Radial density and speed profiles** with bootstrap CIs
4. **Effect sizes (Cohen's d)** with bootstrap CIs
5. **Cross-embedding robustness** quantification
6. **Group flow fields (Rabinovich-style)** with density overlay
7. **Flow field differences** with divergence analysis

**Outputs:**
- `bootstrap_metrics_*.png` - Flow metrics with 95% CIs per group
- `density_diff_ci_*.png` - Density difference maps with significance masking
- `radial_profiles_*.png` - Radial density/speed profiles with CIs
- `group_flow_fields_*.png` - Rabinovich-style flow fields per group
- `flow_difference_*.png` - Flow field differences (MCI-HC, AD-HC)
- `effect_sizes.png` - Cohen's d effect sizes with interpretation
- `cross_embedding_robustness.png` - Consistency heatmaps across embeddings
- `full_analysis_results.json` - All results in machine-readable format
- `summary_table.txt` - Human-readable summary table

**Statistical approach:**
| Analysis | Method | Output |
|----------|--------|--------|
| Flow metrics | Subject-level bootstrap (n=500) | Mean ± 95% CI |
| Density differences | Per-pixel bootstrap | Masked difference maps |
| Radial profiles | Subject-level bootstrap | Profile curves with CI bands |
| Group flow fields | Aggregate displacement vectors | Quiver plots + magnitude maps |
| Flow differences | Vector subtraction | Direction/magnitude/divergence maps |
| Effect sizes | Bootstrap Cohen's d | d ± 95% CI with interpretation |
| Cross-embedding | Spearman correlations | Mean off-diagonal ρ |

**Effect size interpretation:**
| |d| | Interpretation |
|-----|----------------|
| < 0.2 | Negligible |
| 0.2 - 0.5 | Small |
| 0.5 - 0.8 | Medium |
| > 0.8 | Large |

**Theoretical framing:**
This analysis is about *dynamics*, not labels:
- Disease labels are grouping variables, not prediction targets
- The contribution is methodological + conceptual
- Shows that entropy is preserved, flow geometry changes, metastability reorganizes
- Effects are consistent across embeddings (embedding-invariant)
- Effects persist under resampling (bootstrap-robust)

## Output

All plots are saved to:
```
/Users/luki/Documents/GitHub/eeg-state-biomarkers/results/local_analysis/
```

## File Structure

```
scripts/local_analysis/
├── config.py                 # Edit paths here
├── load_model.py             # Model loading utilities
├── load_data.py              # Data loading and preprocessing
├── plot_recurrence.py        # Recurrence matrix visualization
├── plot_umap_3d.py           # 3D UMAP visualization
├── plot_trajectory.py        # Rabinovich-style trajectory visualization
├── compare_hc_mci.py         # Group comparison with violin plots
├── classify_rqa.py           # RQA-based classification with ROC curves
├── multi_embedding.py        # Multi-embedding trajectory analysis
├── full_dataset_analysis.py  # Publication-ready statistical analysis
└── README.md                 # This file
```

## Notes

- Scripts use MPS (Metal Performance Shaders) by default for M1 Mac GPU acceleration
- Change `DEVICE = "cpu"` in config.py if you have issues with MPS
- The model auto-detects whether it's ConvLSTM or Transformer architecture
- Phase data is extracted with circular representation (cos, sin, amplitude)

## Data Organization

The data directory has three clinical groups:

```
data/
├── AD/     # Alzheimer's Disease (label=2)
├── MCI/    # Mild Cognitive Impairment (label=1)
└── HID/    # Healthy Controls (label=0)
```

**Labels:**
- `HC` / `HID` = Healthy Controls (label=0) - **blue** in plots
- `MCI` = Mild Cognitive Impairment (label=1) - **orange** in plots
- `AD` = Alzheimer's Disease (label=2) - **red** in plots

**Default behavior:**
- By default, scripts compare HC vs MCI (no AD subjects)
- Use `--conditions HID MCI AD` to include all three groups
- Use `--conditions HID AD` to compare HC vs AD directly
