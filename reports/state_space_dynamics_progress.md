# State-Space Dynamics Analysis: Progress Report

**Project:** EEG State Biomarkers for MCI Detection
**Date:** January 2026
**Status:** Local Analysis Complete, Awaiting Full Dataset Training

---

## 1. Executive Summary

This report documents progress on analyzing EEG latent-space dynamics using a trained transformer autoencoder. We have developed and validated a multi-embedding analysis pipeline that:

1. **Computes flow geometry metrics** (speed, occupancy entropy, tortuosity) across 5 embedding methods
2. **Validates cross-embedding consistency** to ensure findings are not method artifacts
3. **Compares HC vs MCI groups** with pooled embedding and per-class density aggregation

Key preliminary finding: **Occupancy entropy shows consistent HC > MCI trend across embeddings**, suggesting MCI trajectories are more "trapped" in restricted regions of state space.

---

## 2. Single-Subject Trajectory Analysis

### 2.1 Rabinovich-Style Flow Visualization

The trajectory analysis reveals the flow structure in the learned latent space.

**Flow Field (Quiver Plot):**
![Trajectory Flow](../results/local_analysis/trajectory_flow.png)

The flow field shows local displacement vectors, revealing "channels" and preferred flow directions in the latent space.

**Density Heatmap:**
![Trajectory Density](../results/local_analysis/trajectory_density.png)

Density shows where the system spends most time. Concentrated regions suggest metastable states or attractors.

**Speed-Colored Trajectory:**
![Trajectory Speed](../results/local_analysis/trajectory_speed.png)

Speed coloring identifies slow (metastable) vs fast (transition) regions. Low-speed regions (blue) indicate potential attractor-like states.

**3D Trajectory:**
![Trajectory 3D](../results/local_analysis/trajectory_3d.png)

The 3D view reveals the overall trajectory structure and temporal evolution through the latent space.

---

## 3. Multi-Embedding Analysis

### 3.1 Embedding Methods Compared

| Method | Description | Key Parameters |
|--------|-------------|----------------|
| PCA | Standard linear projection | n_components=2 |
| Time-lagged PCA | Captures temporal structure via lagged covariance | tau=5 samples |
| Diffusion Maps | Nonlinear, respects manifold geometry | k=15 neighbors |
| UMAP | Preserves local + global structure | n_neighbors=15, min_dist=0.1 |
| Delay Embedding | Takens reconstruction of dynamics | tau=5, dim=3 |

### 3.2 HC Subject Example (S017)

![Multi-Embedding HC](../results/local_analysis/multi_embedding_S017.png)

**Embedding Consistency (HC):**
![Embedding Consistency HC](../results/local_analysis/embedding_consistency_S017.png)

### 3.3 MCI Subject Example (i002)

![Multi-Embedding MCI](../results/local_analysis/multi_embedding_i002.png)

**Embedding Consistency (MCI):**
![Embedding Consistency MCI](../results/local_analysis/embedding_consistency_i002.png)

### 3.4 Cross-Embedding Consistency Interpretation

High consistency (Spearman r > 0.7) across embedding methods suggests:
- The underlying dynamics are captured reliably
- Findings are not artifacts of a particular embedding choice
- Flow geometry metrics reflect genuine dynamical structure

---

## 4. Group Comparison: HC vs MCI vs AD

### 4.1 Pooled Embedding Density Maps

The `--compare-all` mode fits each embedding on pooled data from all subjects, then computes per-class density aggregation.

**PCA Density Comparison:**
![PCA Density](../results/local_analysis/compare_all_pca_density.png)

**Time-lagged PCA Density:**
![tPCA Density](../results/local_analysis/compare_all_tpca_density.png)

**Diffusion Maps Density:**
![Diffusion Density](../results/local_analysis/compare_all_diffusion_density.png)

**Delay Embedding Density:**
![Delay Density](../results/local_analysis/compare_all_delay_density.png)

**UMAP Density:**
![UMAP Density](../results/local_analysis/compare_all_umap_density.png)

### 4.2 Flow Metrics by Group

![Flow Metrics Comparison](../results/local_analysis/compare_all_flow_metrics.png)

### 4.3 Key Metric: Occupancy Entropy

| Group | Occupancy Entropy (mean ± std) | Interpretation |
|-------|-------------------------------|----------------|
| HC | Higher | More uniform exploration of state space |
| MCI | Lower | More "trapped" in restricted regions |
| AD | Lowest | Most restricted dynamics |

**Theoretical basis:** Reduced occupancy entropy in MCI/AD suggests:
- Fewer accessible metastable states
- More rigid, less flexible dynamics
- Consistent with "supercritical" brain state hypothesis

---

## 5. RQA-Based Classification

### 5.1 ROC Curve

![ROC Curve](../results/local_analysis/roc_curve.png)

### 5.2 Feature Importance

![Feature Importance](../results/local_analysis/feature_importance.png)

### 5.3 Confusion Matrix

![Confusion Matrix](../results/local_analysis/confusion_matrix.png)

---

## 6. UMAP Visualizations

### 6.1 3D UMAP of Mean Latents

![UMAP Mean Latents](../results/local_analysis/umap_mean_latents_3d.png)

### 6.2 3D UMAP Trajectories

![UMAP Trajectories](../results/local_analysis/umap_trajectories_3d.png)

### 6.3 Per-Subject Coloring

![UMAP Per Subject](../results/local_analysis/umap_per_subject_3d.png)

---

## 7. RQA Metrics Comparison

### 7.1 Violin Plots

![RQA Violin](../results/local_analysis/rqa_violin_comparison.png)

### 7.2 Group Comparison

![RQA Group Comparison](../results/local_analysis/group_comparison_rr2.png)

### 7.3 Expected Trends (Brain Criticality Framework)

| Feature | Description | Expected in MCI |
|---------|-------------|-----------------|
| DET | Determinism (diagonal lines) | ↑ higher |
| LAM | Laminarity (vertical lines) | ↑ higher |
| TT | Trapping Time | ↑ higher |
| L_mean | Mean diagonal length | ↑ higher |
| ENTR | Entropy of diagonals | ↓ lower |
| DIV | Divergence (1/L_max) | ↓ lower |

---

## 8. Recurrence Matrix Examples

![Recurrence Matrices](../results/local_analysis/i007_20150115_1321_recurrence_matrices.png)

---

## 9. Next Steps

### 9.1 Immediate (RunPod GPU)

1. **Train on full dataset** (78 MCI + 31 HC) with modern architecture
2. **Run integration experiment** with all 12 guardrails
3. **Validate occupancy entropy** as primary discriminative metric

### 9.2 Analysis Extensions

1. **Bootstrap confidence intervals** for flow metrics
2. **Permutation testing** for HC vs MCI differences
3. **Cross-validation** of density-based features

### 9.3 Theoretical Development

1. **Quantify "criticality shift"** using RQA + flow metrics jointly
2. **Relate occupancy entropy to attractor dimensionality**
3. **Connect to Rabinovich's IFPS framework** for cognitive dynamics

---

## 10. Technical Notes

### 10.1 Scripts Used

All analysis scripts are in `scripts/local_analysis/`:

```bash
# Single-subject trajectory
python scripts/local_analysis/plot_trajectory.py --subject S017

# Multi-embedding comparison
python scripts/local_analysis/multi_embedding.py --conditions HID MCI

# Pooled group comparison
python scripts/local_analysis/multi_embedding.py --compare-all --conditions HID MCI AD

# RQA classification
python scripts/local_analysis/classify_rqa.py --conditions HID MCI
```

### 10.2 Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| tau (time lag) | 5 samples | ~20ms at 250Hz, captures fast dynamics |
| delay_dim | 3 | Minimal embedding dimension |
| n_neighbors (UMAP) | 15 | Balance local/global structure |
| RR target | 2% | Standard for EEG RQA |

### 10.3 Reproducibility

- UMAP random_state fixed to 42
- All figures saved to `results/local_analysis/`
- Model checkpoint: `models/best.pt`

---

## References

1. Rabinovich, M.I. et al. (2008). Transient Cognitive Dynamics, Metastability, and Decision Making. *PLoS Computational Biology*.
2. Webber, C.L. & Zbilut, J.P. (2005). Recurrence Quantification Analysis of Nonlinear Dynamical Systems. *Tutorials in Contemporary Nonlinear Methods*.
3. Moon, K.R. et al. (2019). Visualizing Structure and Transitions in High-Dimensional Biological Data. *Nature Biotechnology*.

---

*Report generated from local analysis on M1 MacBook. Full dataset training pending on RunPod GPU.*
