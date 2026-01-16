"""State discovery from latent space representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from omegaconf import DictConfig

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

from sklearn.cluster import DBSCAN


@dataclass
class StateAssignment:
    """Result of state discovery."""

    # Per-sample assignments
    state_labels: np.ndarray          # Cluster labels (-1 = noise)
    embedding: np.ndarray             # Reduced embedding (e.g., UMAP)

    # Per-state information
    n_states: int                     # Number of discovered states (excluding noise)
    state_centers: np.ndarray         # Cluster centers in embedding space
    state_sizes: np.ndarray           # Number of samples per state

    # Filtering
    stable_mask: np.ndarray           # Boolean mask for stable states
    noise_mask: np.ndarray            # Boolean mask for noise points


class StateDiscovery:
    """
    Discover latent brain states from autoencoder representations.

    Pipeline:
    1. Dimensionality reduction (UMAP) on latent trajectories
    2. Clustering (DBSCAN or HDBSCAN) to identify states
    3. Filtering based on stability criteria

    IMPORTANT: To prevent label leakage, always fit on training data only
    and use transform() for test data.

    Args:
        cfg: Configuration with state_discovery settings
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # UMAP settings
        umap_cfg = cfg.state_discovery.umap
        if UMAP_AVAILABLE:
            self.reducer = umap.UMAP(
                n_neighbors=umap_cfg.n_neighbors,
                min_dist=umap_cfg.min_dist,
                metric=umap_cfg.metric,
                n_components=umap_cfg.n_components,
                random_state=cfg.experiment.seed,
            )
        else:
            raise ImportError("umap-learn required for state discovery")

        # Clustering settings
        cluster_cfg = cfg.state_discovery.clustering
        if cluster_cfg.method == "hdbscan" and HDBSCAN_AVAILABLE:
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=cluster_cfg.min_cluster_size,
                min_samples=cluster_cfg.min_samples,
            )
        else:
            # Fall back to DBSCAN
            self.clusterer = DBSCAN(
                eps=cluster_cfg.get("eps", 0.5),
                min_samples=cluster_cfg.min_samples,
            )

        # Selection criteria
        self.selection_cfg = cfg.state_discovery.selection

        # Fitted state
        self._is_fitted = False
        self._train_embedding: np.ndarray | None = None
        self._train_labels: np.ndarray | None = None

    def fit(self, latent_trajectories: np.ndarray) -> StateAssignment:
        """
        Fit state discovery on training latent representations.

        Args:
            latent_trajectories: (n_samples, n_timesteps, hidden_dim)
                                 or (n_samples, hidden_dim) if already aggregated

        Returns:
            StateAssignment for training data
        """
        # Aggregate temporal dimension if needed
        if latent_trajectories.ndim == 3:
            # Mean pooling over time
            latent_flat = latent_trajectories.mean(axis=1)
        else:
            latent_flat = latent_trajectories

        # Dimensionality reduction
        embedding = self.reducer.fit_transform(latent_flat)

        # Clustering
        state_labels = self.clusterer.fit_predict(embedding)

        # Store for transform
        self._train_embedding = embedding
        self._train_labels = state_labels
        self._is_fitted = True

        # Create assignment
        return self._create_assignment(embedding, state_labels)

    def transform(self, latent_trajectories: np.ndarray) -> StateAssignment:
        """
        Assign states to new data using fitted model.

        Args:
            latent_trajectories: (n_samples, n_timesteps, hidden_dim)
                                 or (n_samples, hidden_dim)

        Returns:
            StateAssignment for new data
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform()")

        # Aggregate if needed
        if latent_trajectories.ndim == 3:
            latent_flat = latent_trajectories.mean(axis=1)
        else:
            latent_flat = latent_trajectories

        # Transform to embedding space
        embedding = self.reducer.transform(latent_flat)

        # Assign to nearest cluster
        # For HDBSCAN, use approximate_predict if available
        if hasattr(self.clusterer, "approximate_predict"):
            state_labels, _ = hdbscan.approximate_predict(self.clusterer, embedding)
        else:
            # Simple nearest-neighbor assignment
            state_labels = self._assign_to_nearest_cluster(embedding)

        return self._create_assignment(embedding, state_labels)

    def fit_transform(self, latent_trajectories: np.ndarray) -> StateAssignment:
        """Fit and transform in one step."""
        return self.fit(latent_trajectories)

    def _assign_to_nearest_cluster(self, embedding: np.ndarray) -> np.ndarray:
        """Assign points to nearest training cluster."""
        from sklearn.neighbors import NearestNeighbors

        # Get training points that are not noise
        valid_mask = self._train_labels >= 0
        valid_embedding = self._train_embedding[valid_mask]
        valid_labels = self._train_labels[valid_mask]

        if len(valid_embedding) == 0:
            return np.full(len(embedding), -1)

        # Find nearest neighbor
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(valid_embedding)
        distances, indices = nn.kneighbors(embedding)

        # Assign labels
        labels = valid_labels[indices.ravel()]

        # Mark as noise if too far from any cluster
        # (could add distance threshold here)

        return labels

    def _create_assignment(
        self, embedding: np.ndarray, state_labels: np.ndarray
    ) -> StateAssignment:
        """Create StateAssignment from embedding and labels."""
        unique_labels = np.unique(state_labels)
        valid_labels = unique_labels[unique_labels >= 0]
        n_states = len(valid_labels)

        # Compute cluster centers
        centers = []
        sizes = []
        for label in valid_labels:
            mask = state_labels == label
            centers.append(embedding[mask].mean(axis=0))
            sizes.append(mask.sum())

        state_centers = np.array(centers) if centers else np.array([]).reshape(0, embedding.shape[1])
        state_sizes = np.array(sizes)

        # Stability filtering
        stable_mask = self._compute_stability_mask(state_labels, state_sizes)
        noise_mask = state_labels == -1

        return StateAssignment(
            state_labels=state_labels,
            embedding=embedding,
            n_states=n_states,
            state_centers=state_centers,
            state_sizes=state_sizes,
            stable_mask=stable_mask,
            noise_mask=noise_mask,
        )

    def _compute_stability_mask(
        self, state_labels: np.ndarray, state_sizes: np.ndarray
    ) -> np.ndarray:
        """
        Compute mask for samples in stable states.

        Stability criteria from config:
        - Minimum cluster size
        - Exclude transitions (optional)
        """
        # Start with all non-noise as potentially stable
        stable_mask = state_labels >= 0

        # Could add more sophisticated stability checks here
        # e.g., temporal consistency, dwell time

        return stable_mask
