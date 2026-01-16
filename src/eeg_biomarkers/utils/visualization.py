"""Visualization utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_latent_space(
    embedding: np.ndarray,
    labels: np.ndarray,
    label_names: list[str] | None = None,
    title: str = "Latent Space",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot 2D or 3D latent space embedding.

    Args:
        embedding: (n_samples, 2 or 3) embedding coordinates
        labels: (n_samples,) integer labels
        label_names: Names for each label
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    n_dims = embedding.shape[1]
    unique_labels = np.unique(labels)

    if label_names is None:
        label_names = [f"Class {i}" for i in unique_labels]

    if n_dims == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                label=label_names[i],
                alpha=0.6,
                s=20,
            )
        ax.legend()
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")

    elif n_dims == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                embedding[mask, 2],
                label=label_names[i],
                alpha=0.6,
                s=20,
            )
        ax.legend()
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")

    else:
        raise ValueError(f"Embedding must be 2D or 3D, got {n_dims}D")

    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_recurrence_matrix(
    recurrence_matrix: np.ndarray,
    title: str = "Recurrence Matrix",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot a recurrence matrix.

    Args:
        recurrence_matrix: Binary recurrence matrix (N, N)
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(recurrence_matrix, cmap="binary", origin="lower")
    ax.set_xlabel("Time")
    ax.set_ylabel("Time")
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
