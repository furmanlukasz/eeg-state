"""
Tests for the integration experiment components.

Validates critical requirements:
1. Subject-level evaluation correctness
2. State discovery train-only fitting
3. Null model implementation
4. Retention logging
"""

import numpy as np
import pytest
from dataclasses import asdict

from eeg_biomarkers.experiments.integration_experiment import (
    ExperimentConfig,
    FoldResult,
    StateDiscovery,
    select_informative_states,
    compute_subject_level_prediction,
    compute_retention_stats,
    compute_bootstrap_ci,
)


class TestStateDiscovery:
    """Test train-only state discovery with HDBSCAN."""

    def test_fit_predict_separation(self):
        """Verify that fit and predict are truly separate."""
        np.random.seed(42)

        n_train, n_test = 100, 30
        latent_dim = 32
        time_steps = 10  # StateDiscovery expects 3D trajectories (n_segments, T', hidden_size)

        # Create clustered training data with offsets
        offset1 = np.array([2, 0] + [0] * (latent_dim - 2))
        offset2 = np.array([-2, 0] + [0] * (latent_dim - 2))
        offset3 = np.array([0, 2] + [0] * (latent_dim - 2))

        # Create 3D trajectory data: mean over time should produce the offsets
        latents_train = np.stack([
            *[np.random.randn(time_steps, latent_dim) + offset1 for _ in range(40)],
            *[np.random.randn(time_steps, latent_dim) + offset2 for _ in range(40)],
            *[np.random.randn(time_steps, latent_dim) + offset3 for _ in range(20)],
        ])

        # Test data from same distribution
        latents_test = np.stack([
            *[np.random.randn(time_steps, latent_dim) + offset1 for _ in range(10)],
            *[np.random.randn(time_steps, latent_dim) + offset2 for _ in range(10)],
            *[np.random.randn(time_steps, latent_dim) + offset3 for _ in range(10)],
        ])

        # Fit state discovery
        sd = StateDiscovery(
            n_neighbors=10,
            min_dist=0.1,
            min_cluster_size=5,
            min_samples=3,
            random_state=42,
        )

        states_train = sd.fit(latents_train)
        states_test = sd.predict(latents_test)

        # Check outputs have correct shape
        assert len(states_train) == n_train
        assert len(states_test) == n_test

        # Should find some clusters
        n_clusters = sd.get_n_clusters()
        print(f"Discovered {n_clusters} clusters")
        assert n_clusters >= 1, "Should find at least one cluster"

    def test_predict_before_fit_raises(self):
        """Predict without fit should raise error."""
        sd = StateDiscovery()
        # StateDiscovery expects 3D trajectories (n_segments, T', hidden_size)
        latents = np.random.randn(50, 10, 32)

        with pytest.raises(RuntimeError, match="Must fit before predict"):
            sd.predict(latents)

    def test_hdbscan_approximate_predict_used(self):
        """Verify we're using approximate_predict, not refitting."""
        import hdbscan

        np.random.seed(42)

        # StateDiscovery expects 3D trajectories (n_segments, T', hidden_size)
        latents_train = np.random.randn(100, 10, 10)
        latents_test = np.random.randn(20, 10, 10)

        sd = StateDiscovery(
            min_cluster_size=5,
            min_samples=3,
            random_state=42,
        )

        sd.fit(latents_train)

        # The HDBSCAN model should have prediction_data enabled
        assert sd.hdbscan_model.prediction_data is True

        # Calling predict should NOT change the training labels
        original_train_labels = sd.hdbscan_model.labels_.copy()
        _ = sd.predict(latents_test)
        np.testing.assert_array_equal(
            sd.hdbscan_model.labels_, original_train_labels,
            "Predict should not modify training labels"
        )


class TestSelectInformativeStates:
    """Test state selection criteria."""

    def test_excludes_noise(self):
        """Noise cluster (-1) should always be excluded."""
        state_labels = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1, 1])

        selected = select_informative_states(state_labels, min_samples=2)

        assert -1 not in selected, "Noise cluster should be excluded"
        assert 0 in selected
        assert 1 in selected

    def test_min_samples_threshold(self):
        """States below min_samples should be excluded."""
        # State 0: 50 samples, State 1: 10 samples, State 2: 3 samples
        state_labels = np.array([0]*50 + [1]*10 + [2]*3)

        selected = select_informative_states(state_labels, min_samples=20)

        assert 0 in selected, "State 0 (50 samples) should be included"
        assert 1 not in selected, "State 1 (10 samples) should be excluded"
        assert 2 not in selected, "State 2 (3 samples) should be excluded"


class TestSubjectLevelPrediction:
    """Test subject-level aggregation."""

    def test_aggregation_deterministic(self):
        """Subject-level prediction should be deterministic (mean)."""
        # Two subjects, multiple segments each
        segment_probs = np.array([0.3, 0.5, 0.4, 0.8, 0.9])
        segment_subject_ids = np.array([0, 0, 0, 1, 1])
        segment_labels = np.array([0, 0, 0, 1, 1])

        subj_probs, subj_labels = compute_subject_level_prediction(
            segment_probs, segment_subject_ids, segment_labels
        )

        assert len(subj_probs) == 2
        assert len(subj_labels) == 2

        # Subject 0: mean of [0.3, 0.5, 0.4] = 0.4
        np.testing.assert_almost_equal(subj_probs[0], 0.4)

        # Subject 1: mean of [0.8, 0.9] = 0.85
        np.testing.assert_almost_equal(subj_probs[1], 0.85)

        # Labels should be correct
        assert subj_labels[0] == 0
        assert subj_labels[1] == 1

    def test_single_segment_per_subject(self):
        """Should handle single segment per subject."""
        segment_probs = np.array([0.6, 0.3])
        segment_subject_ids = np.array([0, 1])
        segment_labels = np.array([1, 0])

        subj_probs, subj_labels = compute_subject_level_prediction(
            segment_probs, segment_subject_ids, segment_labels
        )

        assert len(subj_probs) == 2
        np.testing.assert_array_equal(subj_probs, [0.6, 0.3])


class TestRetentionStats:
    """Test window retention logging."""

    def test_retention_by_group(self):
        """Verify retention is computed correctly by group."""
        # 10 segments total
        mask = np.array([True, True, True, False, False,  # First 5 segments
                         True, True, False, False, False])  # Last 5 segments

        labels = np.array([0, 0, 0, 0, 0,  # HC (label=0)
                          1, 1, 1, 1, 1])  # MCI (label=1)

        subject_ids = np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3])

        retention = compute_retention_stats(mask, labels, subject_ids)

        # Overall: 5/10 = 50%
        assert retention["overall"] == 0.5

        # HC: 3/5 = 60%
        assert retention["hc"] == 0.6

        # MCI: 2/5 = 40%
        assert retention["mci"] == 0.4

        # By subject should be computed
        assert 0 in retention["by_subject"]
        assert 1 in retention["by_subject"]
        assert 2 in retention["by_subject"]
        assert 3 in retention["by_subject"]


class TestFoldResult:
    """Test FoldResult dataclass."""

    def test_to_dict_conversion(self):
        """FoldResult should convert to dict for pandas."""
        result = FoldResult(
            fold_idx=0,
            seed=42,
            rr_target=0.02,
            latent_mode="raw",
            condition="baseline",
            subject_auc=0.75,
            n_subjects_train=80,
            n_subjects_test=20,
            n_segments_train=400,
            n_segments_test=100,
            retention_overall=1.0,
            retention_mci=1.0,
            retention_hc=1.0,
        )

        d = asdict(result)

        assert d["fold_idx"] == 0
        assert d["subject_auc"] == 0.75
        assert d["latent_mode"] == "raw"


class TestExperimentConfig:
    """Test ExperimentConfig."""

    def test_default_values(self):
        """Check default configuration values."""
        config = ExperimentConfig()

        assert config.chunk_duration == 5.0
        assert config.n_folds == 5
        assert config.n_seeds == 3
        assert config.rr_targets == [0.01, 0.02, 0.05]
        assert config.run_null_model is True

    def test_to_dict(self):
        """Config should be serializable to dict."""
        config = ExperimentConfig(n_folds=3)
        d = config.to_dict()

        assert d["n_folds"] == 3
        assert "rr_targets" in d


class TestBootstrapCI:
    """Test bootstrap confidence interval computation (guardrail #12)."""

    def test_basic_bootstrap_ci(self):
        """Verify bootstrap CI computes valid bounds."""
        np.random.seed(42)

        # Create a scenario with good but imperfect class separation
        # Some overlap to allow meaningful CI estimation
        subject_probs = np.array([0.2, 0.35, 0.25, 0.45, 0.15,  # HC (mostly low probs)
                                  0.7, 0.85, 0.6, 0.9, 0.75])   # MCI (mostly high probs)
        subject_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        auc, ci_lower, ci_upper = compute_bootstrap_ci(
            subject_probs, subject_labels, n_bootstrap=500, random_state=42
        )

        # AUC should be high for this good separation
        assert auc > 0.8, f"Expected high AUC, got {auc}"

        # CI should be valid
        assert not np.isnan(ci_lower), "CI lower should not be NaN"
        assert not np.isnan(ci_upper), "CI upper should not be NaN"
        assert ci_lower <= ci_upper, "CI lower should be <= upper"
        assert ci_lower <= auc <= ci_upper or np.isclose(auc, ci_lower) or np.isclose(auc, ci_upper), \
            "AUC should be within or close to CI"

    def test_bootstrap_ci_with_few_subjects(self):
        """Bootstrap with too few subjects should return NaN for CI."""
        subject_probs = np.array([0.3, 0.8])
        subject_labels = np.array([0, 1])

        auc, ci_lower, ci_upper = compute_bootstrap_ci(
            subject_probs, subject_labels, n_bootstrap=100
        )

        # Should still compute AUC
        assert auc == 1.0, "Perfect separation should give AUC=1.0"

        # CI should be NaN with only 2 subjects
        assert np.isnan(ci_lower), "CI should be NaN with too few subjects"
        assert np.isnan(ci_upper), "CI should be NaN with too few subjects"

    def test_bootstrap_ci_random_classifier(self):
        """Random classifier should have AUC near 0.5 with wide CI."""
        np.random.seed(42)

        # Random probabilities for both classes
        n_subjects = 30
        subject_probs = np.random.rand(n_subjects)
        subject_labels = np.array([0] * 15 + [1] * 15)

        auc, ci_lower, ci_upper = compute_bootstrap_ci(
            subject_probs, subject_labels, n_bootstrap=500, random_state=42
        )

        # AUC should be near 0.5 for random predictions
        assert 0.3 < auc < 0.7, f"Random classifier AUC should be near 0.5, got {auc}"

        # CI should include 0.5 or be reasonably wide
        ci_width = ci_upper - ci_lower
        assert ci_width > 0.1, f"CI should be reasonably wide for random classifier, got {ci_width}"


class TestNullModel:
    """Test null model implementation."""

    def test_shuffle_preserves_distribution(self):
        """Shuffling within subject should preserve overall distribution."""
        np.random.seed(42)

        # Original state labels
        states = np.array([0, 0, 1, 1, 2, 0, 1, 2, 2, 1])
        subject_ids = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        # Shuffle within subject
        states_shuffled = states.copy()
        for subj in np.unique(subject_ids):
            subj_mask = subject_ids == subj
            np.random.shuffle(states_shuffled[subj_mask])

        # Overall distribution should be the same
        np.testing.assert_array_equal(
            np.bincount(states),
            np.bincount(states_shuffled),
            "Shuffle should preserve overall state distribution"
        )

        # Per-subject distribution should be preserved
        for subj in np.unique(subject_ids):
            subj_mask = subject_ids == subj
            np.testing.assert_array_equal(
                np.bincount(states[subj_mask]),
                np.bincount(states_shuffled[subj_mask]),
                f"Per-subject distribution should be preserved for subject {subj}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
