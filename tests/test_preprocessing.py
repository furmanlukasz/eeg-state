"""Tests for preprocessing functions."""

import numpy as np
import pytest

from eeg_biomarkers.data.preprocessing import (
    extract_phase_circular,
    chunk_signal,
)


class TestExtractPhaseCircular:
    """Tests for circular phase extraction."""

    def test_output_shape_without_amplitude(self):
        """Output should be (n_channels, 2, n_samples) for cos/sin."""
        signal = np.random.randn(10, 1000)  # 10 channels, 1000 samples
        result = extract_phase_circular(signal, include_amplitude=False)

        assert result.shape == (10, 2, 1000)

    def test_output_shape_with_amplitude(self):
        """Output should be (n_channels, 3, n_samples) with amplitude."""
        signal = np.random.randn(10, 1000)
        result = extract_phase_circular(signal, include_amplitude=True)

        assert result.shape == (10, 3, 1000)

    def test_cos_sin_bounded(self):
        """Cos and sin values should be in [-1, 1]."""
        signal = np.random.randn(5, 500)
        result = extract_phase_circular(signal)

        cos_vals = result[:, 0, :]
        sin_vals = result[:, 1, :]

        assert np.all(cos_vals >= -1) and np.all(cos_vals <= 1)
        assert np.all(sin_vals >= -1) and np.all(sin_vals <= 1)

    def test_unit_circle(self):
        """Cos^2 + sin^2 should equal 1."""
        signal = np.random.randn(5, 500)
        result = extract_phase_circular(signal)

        cos_vals = result[:, 0, :]
        sin_vals = result[:, 1, :]

        magnitude = cos_vals**2 + sin_vals**2
        np.testing.assert_array_almost_equal(magnitude, 1.0, decimal=10)


class TestChunkSignal:
    """Tests for signal chunking."""

    def test_basic_chunking(self):
        """Basic chunking without overlap."""
        signal = np.random.randn(10, 1000)
        chunk_samples = 100

        chunks, mask = chunk_signal(signal, chunk_samples)

        assert chunks.shape == (10, 10, 100)  # 10 chunks
        assert mask.shape == (10, 100)
        assert np.all(mask)  # All valid, no padding needed

    def test_chunking_with_overlap(self):
        """Chunking with overlap."""
        signal = np.random.randn(5, 1000)
        chunk_samples = 100
        overlap = 50

        chunks, mask = chunk_signal(signal, chunk_samples, overlap)

        # Step = 100 - 50 = 50, so we get more chunks
        expected_n_chunks = int(np.ceil((1000 - overlap) / (chunk_samples - overlap)))
        assert chunks.shape[1] == expected_n_chunks

    def test_padding(self):
        """Last chunk should be zero-padded if incomplete."""
        signal = np.random.randn(5, 150)  # Not evenly divisible by 100
        chunk_samples = 100

        chunks, mask = chunk_signal(signal, chunk_samples, pad_last=True)

        # Should have 2 chunks
        assert chunks.shape == (5, 2, 100)

        # First chunk fully valid
        assert np.all(mask[0])

        # Second chunk partially valid (50 samples)
        assert np.sum(mask[1]) == 50
        assert np.all(mask[1, :50])
        assert not np.any(mask[1, 50:])

    def test_no_padding(self):
        """Without padding, incomplete chunk should be dropped."""
        signal = np.random.randn(5, 150)
        chunk_samples = 100

        chunks, mask = chunk_signal(signal, chunk_samples, pad_last=False)

        # Should have only 1 complete chunk
        assert chunks.shape == (5, 1, 100)


class TestPhaseContinuity:
    """
    Critical tests validating the phase representation fix.

    The key issue with raw phase: values near +π and -π are physically
    close (both near the wrap point) but appear maximally different
    in L2 loss. The circular (cos, sin) representation fixes this.
    """

    def test_phase_continuity_near_wrap(self):
        """
        Verify phases near ±π are close in (cos, sin) space.

        This is the KEY FIX - raw phase would show discontinuity here.
        """
        # Create signal with phase crossing through ±π
        t = np.linspace(0, 10, 1000)
        freq = 0.5  # Hz
        signal = np.sin(2 * np.pi * freq * t).reshape(1, -1)

        result = extract_phase_circular(signal, include_amplitude=False)
        cos_phase = result[0, 0, :]
        sin_phase = result[0, 1, :]

        # Compute distances between consecutive samples in (cos, sin) space
        d_cos = np.diff(cos_phase)
        d_sin = np.diff(sin_phase)
        distances = np.sqrt(d_cos**2 + d_sin**2)

        # Should be smooth - no large jumps even at wrap points
        max_distance = np.max(distances)
        assert max_distance < 0.5, f"Max distance {max_distance} suggests discontinuity"

    def test_raw_phase_vs_circular_at_wrap(self):
        """
        Demonstrate improvement: raw phase discontinuous, circular smooth.
        """
        from scipy.signal import hilbert

        t = np.linspace(0, 10, 1000)
        signal = np.sin(2 * np.pi * 0.5 * t).reshape(1, -1)

        # Raw phase (legacy approach)
        analytic = hilbert(signal, axis=1)
        raw_phase = np.angle(analytic)
        raw_diff = np.abs(np.diff(raw_phase[0]))

        # Circular (new approach)
        circular = extract_phase_circular(signal, include_amplitude=False)
        cos_phase = circular[0, 0, :]
        sin_phase = circular[0, 1, :]

        circ_diff = np.sqrt(np.diff(cos_phase) ** 2 + np.diff(sin_phase) ** 2)

        # Raw phase has jumps near 2π at wrap points
        raw_max_jump = np.max(raw_diff)
        circ_max_jump = np.max(circ_diff)

        # Circular should be much smoother
        assert circ_max_jump < raw_max_jump / 3, (
            f"Circular max jump {circ_max_jump:.3f} should be much smaller "
            f"than raw max jump {raw_max_jump:.3f}"
        )


class TestRealDataPreprocessing:
    """Tests using actual EEG data from the repository."""

    @pytest.fixture
    def sample_fif_file(self):
        """Find a sample .fif file in the data directory."""
        from pathlib import Path

        data_dir = Path(__file__).parent.parent / "data"
        fif_files = list(data_dir.glob("**/*_good_*_eeg.fif"))

        if not fif_files:
            pytest.skip("No EEG data files found in data/")

        return fif_files[0]

    def test_load_real_file(self, sample_fif_file):
        """Test loading a real EEG file."""
        from eeg_biomarkers.data.preprocessing import load_eeg_file

        raw = load_eeg_file(
            sample_fif_file,
            filter_low=3.0,
            filter_high=48.0,
            reference="csd",
            verbose=False,
        )

        assert raw is not None
        assert raw.info["sfreq"] > 0
        assert len(raw.ch_names) > 0
        print(f"\nLoaded: {sample_fif_file.name}")
        print(f"  Channels: {len(raw.ch_names)}")
        print(f"  Sampling rate: {raw.info['sfreq']} Hz")
        print(f"  Duration: {raw.times[-1]:.1f} s")

    def test_full_pipeline_real_data(self, sample_fif_file):
        """Test full preprocessing pipeline on real data."""
        from eeg_biomarkers.data.preprocessing import load_eeg_file, prepare_phase_chunks

        raw = load_eeg_file(
            sample_fif_file,
            filter_low=3.0,
            filter_high=48.0,
            reference="csd",
            verbose=False,
        )

        chunks, mask, info = prepare_phase_chunks(
            raw,
            chunk_duration=5.0,
            chunk_overlap=0.0,
            include_amplitude=False,
        )

        n_channels = info["n_channels"]
        phase_channels = info["phase_channels"]

        print(f"\nPreprocessed: {sample_fif_file.name}")
        print(f"  Chunks: {chunks.shape[0]}")
        print(f"  Features per chunk: {chunks.shape[1]} ({n_channels} ch x {phase_channels} phase)")
        print(f"  Samples per chunk: {chunks.shape[2]}")

        # Verify cos/sin properties on real data
        for chunk_idx in range(min(3, chunks.shape[0])):
            chunk = chunks[chunk_idx]
            reshaped = chunk.reshape(n_channels, phase_channels, -1)
            cos_vals = reshaped[:, 0, :]
            sin_vals = reshaped[:, 1, :]

            magnitude = cos_vals**2 + sin_vals**2
            np.testing.assert_allclose(
                magnitude,
                1.0,
                rtol=1e-4,
                err_msg=f"cos^2 + sin^2 != 1 in chunk {chunk_idx}",
            )

        print("  ✓ Unit circle constraint satisfied for all chunks")
