"""
Tests for artifact control and HF-proxy analysis.

These tests validate the critic agent's recommended checks:
1. Proper HF power computation (not crude signal variance)
2. Topographic analysis (EMG-prone vs central electrodes)
3. Latent residualization effectiveness
4. State-HF relationship testing
"""

import numpy as np
import pytest
from scipy import signal

from eeg_biomarkers.analysis.artifact_control import (
    compute_hf_power,
    analyze_latent_hf_correlation,
    residualize_latent,
    check_states_vs_hf_proxy,
    run_artifact_control_analysis,
)


class TestHFPowerComputation:
    """Test proper HF power computation."""

    def test_welch_method_basic(self):
        """Test Welch PSD-based HF power computation."""
        np.random.seed(42)

        # Create synthetic EEG with known HF content
        sfreq = 250.0
        n_channels = 64
        n_times = 1250  # 5 seconds
        n_segments = 10

        # Generate signal with varying HF power across segments
        data = np.zeros((n_segments, n_channels, n_times))
        expected_hf = np.zeros(n_segments)

        for seg in range(n_segments):
            # Base signal (low freq)
            t = np.arange(n_times) / sfreq
            base = np.sin(2 * np.pi * 10 * t)  # 10 Hz

            # Add HF noise with increasing power
            hf_amplitude = 0.1 + seg * 0.1
            hf_noise = hf_amplitude * np.random.randn(n_times)
            # Bandpass to 40-100 Hz
            sos = signal.butter(4, [40, 100], btype='band', fs=sfreq, output='sos')
            hf_noise = signal.sosfiltfilt(sos, hf_noise)

            for ch in range(n_channels):
                data[seg, ch] = base + hf_noise + np.random.randn(n_times) * 0.01

            expected_hf[seg] = hf_amplitude

        # Compute HF power
        result = compute_hf_power(data, sfreq, band=(40, 100), method="welch")

        # Check output shape
        assert result.hf_power.shape == (n_segments,)

        # HF power should correlate with our injected amplitude
        corr = np.corrcoef(result.hf_power, expected_hf)[0, 1]
        print(f"\nCorrelation between computed and expected HF power: {corr:.3f}")
        assert corr > 0.8, f"HF power should track injected amplitude (r={corr:.3f})"

    def test_topographic_analysis(self):
        """Test topographic breakdown by electrode region."""
        np.random.seed(42)

        sfreq = 250.0
        n_times = 1250
        n_segments = 5

        # Define channel names with topographic info
        channel_names = [
            # Temporal (EMG-prone)
            "T7", "T8", "TP9", "TP10", "FT7", "FT8",
            # Central (more neural)
            "Cz", "C3", "C4", "CPz",
            # Occipital (most neural)
            "O1", "O2", "Oz", "POz",
        ]
        n_channels = len(channel_names)

        # Create data with EMG-like pattern: high HF in temporal, low elsewhere
        data = np.random.randn(n_segments, n_channels, n_times) * 0.1

        # Add strong HF to temporal channels only
        temporal_idx = [0, 1, 2, 3, 4, 5]
        for seg in range(n_segments):
            for idx in temporal_idx:
                hf_noise = np.random.randn(n_times)
                sos = signal.butter(4, [40, 100], btype='band', fs=sfreq, output='sos')
                data[seg, idx] += signal.sosfiltfilt(sos, hf_noise) * 2.0

        result = compute_hf_power(data, sfreq, band=(40, 100), channel_names=channel_names)

        # Check topographic breakdown exists
        assert result.hf_power_temporal is not None
        assert result.hf_power_central is not None
        assert result.hf_power_occipital is not None

        # Temporal should have highest HF power (EMG pattern)
        print(f"\nHF power by region (mean across segments):")
        print(f"  Temporal: {result.hf_power_temporal.mean():.4f}")
        print(f"  Central:  {result.hf_power_central.mean():.4f}")
        print(f"  Occipital: {result.hf_power_occipital.mean():.4f}")

        assert result.hf_power_temporal.mean() > result.hf_power_central.mean(), \
            "Temporal should have higher HF than central (EMG pattern)"
        assert result.hf_power_temporal.mean() > result.hf_power_occipital.mean(), \
            "Temporal should have higher HF than occipital (EMG pattern)"

    def test_nyquist_warning(self):
        """Test that band exceeding Nyquist is handled."""
        np.random.seed(42)

        sfreq = 100.0  # Low sampling rate
        data = np.random.randn(5, 32, 500)

        # Band extends beyond Nyquist (50 Hz)
        with pytest.warns(UserWarning, match="exceeds Nyquist"):
            result = compute_hf_power(data, sfreq, band=(40, 100))

        # Should still return valid results
        assert result.hf_power.shape == (5,)


class TestLatentHFCorrelation:
    """Test latent-HF correlation analysis."""

    def test_correlation_analysis_independent(self):
        """Test analysis when latent and HF are independent."""
        np.random.seed(42)

        n_segments = 100
        latent_dim = 32

        # Independent latent and HF proxy
        latents = np.random.randn(n_segments, latent_dim)
        hf_power = np.random.randn(n_segments)

        result = analyze_latent_hf_correlation(latents, hf_power)

        print(f"\nIndependent case:")
        print(f"  Correlation with norm: {result.correlation_with_latent_norm:.3f}")
        print(f"  Likely artifact: {result.likely_artifact}")

        # Should have low correlation
        assert abs(result.correlation_with_latent_norm) < 0.3
        assert not result.likely_artifact

    def test_correlation_analysis_correlated(self):
        """Test analysis when latent correlates with HF."""
        np.random.seed(42)

        n_segments = 100
        latent_dim = 32

        # Create correlated latent
        hf_power = np.random.randn(n_segments)
        latents = np.random.randn(n_segments, latent_dim)
        # Make first dimension correlate with HF
        latents[:, 0] = hf_power * 2 + np.random.randn(n_segments) * 0.5

        result = analyze_latent_hf_correlation(latents, hf_power)

        print(f"\nCorrelated case:")
        print(f"  Correlation with norm: {result.correlation_with_latent_norm:.3f}")
        print(f"  Max per-dim correlation: {result.max_dim_correlation:.3f} (dim {result.max_dim_index})")

        # Should detect the correlation
        assert result.max_dim_index == 0, "Should identify dim 0 as most correlated"
        assert abs(result.max_dim_correlation) > 0.8


class TestResidualizeLatent:
    """Test latent residualization."""

    def test_residualization_removes_correlation(self):
        """Test that residualization removes HF-proxy influence."""
        np.random.seed(42)

        n_train = 80
        n_test = 20
        latent_dim = 32

        # Create strongly correlated data
        hf_train = np.random.randn(n_train)
        hf_test = np.random.randn(n_test)

        # Latent with HF-correlated component
        latents_train = np.random.randn(n_train, latent_dim)
        latents_test = np.random.randn(n_test, latent_dim)

        # Inject HF correlation
        latents_train[:, 0] = hf_train * 2 + np.random.randn(n_train) * 0.3
        latents_test[:, 0] = hf_test * 2 + np.random.randn(n_test) * 0.3

        # Check pre-residualization correlation
        pre_corr = np.corrcoef(hf_train, latents_train[:, 0])[0, 1]
        print(f"\nPre-residualization correlation: {pre_corr:.3f}")

        # Residualize
        resid_train, resid_test, reg = residualize_latent(
            latents_train, latents_test, hf_train, hf_test
        )

        # Check post-residualization
        post_corr_train = np.corrcoef(hf_train, resid_train[:, 0])[0, 1]
        post_corr_test = np.corrcoef(hf_test, resid_test[:, 0])[0, 1]

        print(f"Post-residualization train correlation: {post_corr_train:.3f}")
        print(f"Post-residualization test correlation: {post_corr_test:.3f}")

        # Correlation should be greatly reduced
        assert abs(post_corr_train) < abs(pre_corr) * 0.3, \
            "Residualization should reduce correlation"

    def test_residualization_train_only_fitting(self):
        """Verify regression is fit on train only (no leakage)."""
        np.random.seed(42)

        n_train = 50
        n_test = 50
        latent_dim = 16

        # Train and test have different HF-latent relationships
        hf_train = np.random.randn(n_train)
        hf_test = np.random.randn(n_test) + 5  # Shifted

        latents_train = np.outer(hf_train, np.ones(latent_dim)) * 2
        latents_test = np.outer(hf_test, np.ones(latent_dim)) * 3  # Different slope

        resid_train, resid_test, reg = residualize_latent(
            latents_train, latents_test, hf_train, hf_test
        )

        # The regression should be fit to train slope (2), not test slope (3)
        # So residual test should still have some correlation
        # This is the correct behavior - we don't want to fit on test

        train_residual_mean = np.abs(resid_train).mean()
        test_residual_mean = np.abs(resid_test).mean()

        print(f"\nTrain residual mean: {train_residual_mean:.3f}")
        print(f"Test residual mean: {test_residual_mean:.3f}")

        # Train residuals should be smaller (model fit to train)
        assert train_residual_mean < test_residual_mean, \
            "Train residuals should be smaller (model fit on train only)"


class TestStatesVsHFProxy:
    """Test state-HF relationship testing."""

    def test_independent_states(self):
        """Test when states are independent of HF proxy."""
        np.random.seed(42)

        n_samples = 100
        n_states = 3

        # Random states
        state_labels = np.random.randint(0, n_states, n_samples)
        hf_proxy = np.random.randn(n_samples)

        result = check_states_vs_hf_proxy(state_labels, hf_proxy)

        print(f"\nIndependent states test:")
        print(f"  Kruskal-Wallis p-value: {result['kruskal_pvalue']:.4f}")
        print(f"  Prediction accuracy: {result['prediction_accuracy']:.3f}")
        print(f"  Interpretation: {result['interpretation']}")

        # Should not be able to predict state from HF
        assert result["kruskal_pvalue"] > 0.01, \
            "p-value should be high for independent states"

    def test_hf_driven_states(self):
        """Test when states are essentially HF-proxy bins."""
        np.random.seed(42)

        n_samples = 100

        # Create HF proxy
        hf_proxy = np.random.randn(n_samples)

        # States are just HF bins
        state_labels = np.digitize(hf_proxy, bins=[-1, 0, 1]) - 1

        result = check_states_vs_hf_proxy(state_labels, hf_proxy)

        print(f"\nHF-driven states test:")
        print(f"  Kruskal-Wallis p-value: {result['kruskal_pvalue']:.4f}")
        print(f"  Prediction accuracy: {result['prediction_accuracy']:.3f}")
        print(f"  Interpretation: {result['interpretation']}")

        # Should easily predict state from HF
        assert result["kruskal_pvalue"] < 0.001, \
            "p-value should be very low for HF-driven states"
        assert "WARNING" in result["interpretation"], \
            "Should warn about HF-state relationship"


class TestFullPipeline:
    """Test the complete artifact control pipeline."""

    def test_run_artifact_control_analysis(self):
        """Test full analysis pipeline."""
        np.random.seed(42)

        n_segments = 50
        n_channels = 32
        n_times = 1250
        latent_dim = 64
        sfreq = 250.0

        # Generate synthetic data
        raw_data = np.random.randn(n_segments, n_channels, n_times)
        latents = np.random.randn(n_segments, latent_dim)

        # Run full analysis
        results = run_artifact_control_analysis(
            latents=latents,
            raw_data=raw_data,
            sfreq=sfreq,
            channel_names=None,  # No topographic analysis
            state_labels=None,   # No state analysis
            hf_band=(40, 100),
        )

        # Check all components are present
        assert "hf_proxy" in results
        assert "artifact_analysis" in results
        assert "residual_latent" in results
        assert "summary" in results

        print(f"\n=== Full Pipeline Results ===")
        print(f"Original correlation: {results['summary']['original_correlation']:.3f}")
        print(f"Post-residualization: {results['summary']['post_residualization_correlation']:.3f}")
        print(f"Recommendation: {results['summary']['recommendation']}")

        # Residualization should reduce correlation
        assert abs(results["summary"]["post_residualization_correlation"]) <= \
               abs(results["summary"]["original_correlation"]) + 0.1


class TestFoldDiagnostics:
    """Test fold diagnostic reporting functions."""

    def test_compute_fold_diagnostics(self):
        """Test diagnostic computation for a single fold."""
        from eeg_biomarkers.analysis.artifact_control import (
            compute_fold_diagnostics,
            FoldDiagnosticReport,
        )

        np.random.seed(42)

        n_train = 40
        n_test = 10
        n_channels = 32
        n_times = 1250
        latent_dim = 64
        sfreq = 250.0

        # Generate synthetic data
        raw_train = np.random.randn(n_train, n_channels, n_times)
        raw_test = np.random.randn(n_test, n_channels, n_times)
        latents_train = np.random.randn(n_train, latent_dim)
        latents_test = np.random.randn(n_test, latent_dim)

        # Subject IDs
        train_subjects = np.repeat(np.arange(8), 5)  # 8 subjects, 5 segments each
        test_subjects = np.repeat(np.arange(8, 10), 5)  # 2 subjects

        # State labels for training
        state_labels = np.random.randint(0, 3, n_train)

        report = compute_fold_diagnostics(
            raw_data_train=raw_train,
            raw_data_test=raw_test,
            latents_train=latents_train,
            latents_test=latents_test,
            sfreq=sfreq,
            fold_idx=0,
            channel_names=None,
            state_labels_train=state_labels,
            train_subject_ids=train_subjects,
            test_subject_ids=test_subjects,
        )

        # Check return type
        assert isinstance(report, FoldDiagnosticReport)

        # Check fields are populated
        assert report.fold_idx == 0
        assert report.n_train_segments == n_train
        assert report.n_test_segments == n_test
        assert report.n_train_subjects == 8
        assert report.n_test_subjects == 2

        # HF correlations should be computed
        assert report.hf1_correlation_raw is not None
        assert report.hf1_correlation_residualized is not None
        assert report.hf2_correlation_raw is not None  # sfreq=250 allows 70-110 Hz

        # State prediction should be computed
        assert report.hf_state_prediction_accuracy is not None
        assert report.hf_state_kruskal_pvalue is not None

        # Classification metrics should be None (filled by caller)
        assert report.baseline_auc is None
        assert report.state_conditioned_auc is None

        print(f"\nFold 0 diagnostics:")
        print(f"  HF1 raw correlation: {report.hf1_correlation_raw:.3f}")
        print(f"  HF1 residualized: {report.hf1_correlation_residualized:.3f}")
        print(f"  HF2 raw correlation: {report.hf2_correlation_raw:.3f}")
        print(f"  State pred accuracy: {report.hf_state_prediction_accuracy:.3f}")

    def test_format_diagnostics_summary(self):
        """Test summary formatting for multiple folds."""
        from eeg_biomarkers.analysis.artifact_control import (
            FoldDiagnosticReport,
            format_diagnostics_summary,
        )

        # Create mock reports
        reports = []
        for i in range(5):
            report = FoldDiagnosticReport(
                fold_idx=i,
                hf1_correlation_raw=0.3 + np.random.randn() * 0.1,
                hf2_correlation_raw=0.2 + np.random.randn() * 0.1,
                hf1_correlation_residualized=0.05 + np.random.randn() * 0.05,
                hf2_correlation_residualized=0.03 + np.random.randn() * 0.05,
                hf1_temporal_central_ratio=1.2 + np.random.randn() * 0.2,
                hf2_temporal_central_ratio=1.4 + np.random.randn() * 0.2,
                hf_state_prediction_accuracy=0.4 + np.random.randn() * 0.1,
                hf_state_kruskal_pvalue=0.1 + np.random.rand() * 0.3,
                baseline_auc=0.65 + np.random.randn() * 0.05,
                state_conditioned_auc=0.72 + np.random.randn() * 0.05,
                raw_latent_auc=0.68 + np.random.randn() * 0.05,
                residualized_latent_auc=0.67 + np.random.randn() * 0.05,
                n_train_segments=100,
                n_test_segments=25,
                n_train_subjects=8,
                n_test_subjects=2,
            )
            reports.append(report)

        summary = format_diagnostics_summary(reports)

        # Check that summary contains expected sections
        assert "PRE-INTEGRATION DIAGNOSTIC SUMMARY" in summary
        assert "HF1 (30-48 Hz)" in summary
        assert "HF2 (70-110 Hz)" in summary
        assert "Topographic Analysis" in summary
        assert "Classification Performance" in summary
        assert "State conditioning effect" in summary

        print(f"\n{summary}")

    def test_fold_report_to_dict(self):
        """Test conversion to dictionary for serialization."""
        from eeg_biomarkers.analysis.artifact_control import FoldDiagnosticReport

        report = FoldDiagnosticReport(
            fold_idx=0,
            hf1_correlation_raw=0.35,
            hf2_correlation_raw=0.25,
            hf1_correlation_residualized=0.05,
            hf2_correlation_residualized=0.03,
            hf1_temporal_central_ratio=1.1,
            hf2_temporal_central_ratio=1.3,
            hf_state_prediction_accuracy=0.4,
            hf_state_kruskal_pvalue=0.15,
            baseline_auc=0.65,
            state_conditioned_auc=0.72,
            raw_latent_auc=0.68,
            residualized_latent_auc=0.67,
            n_train_segments=100,
            n_test_segments=25,
            n_train_subjects=8,
            n_test_subjects=2,
        )

        d = report.to_dict()

        assert d["fold_idx"] == 0
        assert d["hf1_correlation_raw"] == 0.35
        assert d["baseline_auc"] == 0.65
        assert d["n_train_subjects"] == 8

        # Should be JSON serializable
        import json
        json_str = json.dumps(d)
        assert "hf1_correlation_raw" in json_str


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
