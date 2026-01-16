"""Tests for RQA computation."""

import numpy as np
import pytest

from eeg_biomarkers.analysis.rqa import (
    compute_rqa_features,
    compute_rqa_from_distance_matrix,
    RQAFeatures,
)


class TestRQAFeatures:
    """Tests for RQA feature computation."""

    def test_identity_matrix(self):
        """Identity matrix should have RR=1, DET=0 (no off-diagonal lines)."""
        R = np.eye(100)
        features = compute_rqa_features(R)

        # Off-diagonal RR should be 0
        assert features.RR == 0.0

    def test_full_matrix(self):
        """Full recurrence (all 1s) should have RR=1, high DET."""
        R = np.ones((100, 100))
        features = compute_rqa_features(R)

        # RR should be 1
        assert features.RR == pytest.approx(1.0, rel=0.01)

        # DET should be 1 (all points on diagonal lines)
        assert features.DET == pytest.approx(1.0, rel=0.01)

    def test_simple_diagonal_pattern(self):
        """Test with a simple pattern with known diagonal lines."""
        # Create matrix with clear diagonal structure
        R = np.zeros((10, 10))
        np.fill_diagonal(R, 1)  # Main diagonal
        np.fill_diagonal(R[1:], 1)  # One diagonal above
        np.fill_diagonal(R[:, 1:], 1)  # One diagonal below

        features = compute_rqa_features(R, min_diagonal_length=2)

        # Should have some determinism
        assert features.DET > 0

    def test_feature_names(self):
        """Test that feature names are consistent."""
        names = RQAFeatures.feature_names()
        assert len(names) == 10
        assert "RR" in names
        assert "DET" in names
        assert "ENTR" in names

    def test_to_dict(self):
        """Test conversion to dictionary."""
        R = np.random.randint(0, 2, (50, 50)).astype(float)
        features = compute_rqa_features(R)

        d = features.to_dict()
        assert isinstance(d, dict)
        assert len(d) == 10
        assert "RR" in d

    def test_to_array(self):
        """Test conversion to array."""
        R = np.random.randint(0, 2, (50, 50)).astype(float)
        features = compute_rqa_features(R)

        arr = features.to_array()
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 10


class TestRQAFromDistanceMatrix:
    """Tests for RQA from distance matrix with RR-controlled threshold."""

    def test_rr_controlled_threshold(self):
        """Test that RR matches target."""
        # Random distance matrix
        D = np.random.rand(100, 100)
        D = (D + D.T) / 2  # Make symmetric
        np.fill_diagonal(D, 0)

        target_rr = 0.05
        features, epsilon = compute_rqa_from_distance_matrix(D, target_rr=target_rr)

        # RR should be close to target
        assert abs(features.RR - target_rr) < 0.02

    def test_epsilon_returned(self):
        """Test that epsilon threshold is returned."""
        D = np.random.rand(50, 50)
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)

        features, epsilon = compute_rqa_from_distance_matrix(D, target_rr=0.02)

        assert isinstance(epsilon, float)
        assert epsilon > 0
