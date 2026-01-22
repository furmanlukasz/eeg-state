"""Tests for dataset configuration system."""

import pytest
from pathlib import Path
import tempfile
import os

from eeg_biomarkers.data.dataset_config import (
    DatasetConfig,
    GroupConfig,
    PreprocessingConfig,
    GreekRestingConfig,
    MeditationBIDSConfig,
    get_dataset_config,
    list_datasets,
    register_dataset,
    DATASET_REGISTRY,
    dataset_from_hydra_config,
)


class TestGroupConfig:
    """Tests for GroupConfig dataclass."""

    def test_basic_creation(self):
        """Test creating a basic GroupConfig."""
        group = GroupConfig(name="HC", label=0, path="HC-RAW/FILT")
        assert group.name == "HC"
        assert group.label == 0
        assert group.path == "HC-RAW/FILT"

    def test_negative_label_raises(self):
        """Test that negative labels raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            GroupConfig(name="invalid", label=-1, path="path")


class TestPreprocessingConfig:
    """Tests for PreprocessingConfig dataclass."""

    def test_defaults(self):
        """Test default preprocessing values."""
        prep = PreprocessingConfig()
        assert prep.filter_low == 1.0
        assert prep.filter_high == 48.0
        assert prep.notch_freq is None
        assert prep.reference == "average"
        assert prep.chunk_duration == 5.0

    def test_custom_values(self):
        """Test custom preprocessing values."""
        prep = PreprocessingConfig(
            filter_low=2.0,
            filter_high=100.0,
            notch_freq=50.0,
            reference="csd",
            chunk_duration=10.0,
        )
        assert prep.filter_low == 2.0
        assert prep.filter_high == 100.0
        assert prep.notch_freq == 50.0
        assert prep.reference == "csd"
        assert prep.chunk_duration == 10.0


class TestGreekRestingConfig:
    """Tests for GreekRestingConfig."""

    def test_defaults(self):
        """Test default Greek dataset configuration."""
        config = GreekRestingConfig()
        assert config.name == "greek_resting"
        assert config.paradigm == "resting"
        assert len(config.groups) == 3  # HC, MCI, AD
        assert config.file_pattern == "*_eeg.fif"

    def test_get_label_map(self):
        """Test label map generation."""
        config = GreekRestingConfig()
        label_map = config.get_label_map()
        assert label_map == {"HC": 0, "MCI": 1, "AD": 2}

    def test_get_label_name(self):
        """Test label name lookup."""
        config = GreekRestingConfig()
        assert config.get_label_name(0) == "HC"
        assert config.get_label_name(1) == "MCI"
        assert config.get_label_name(2) == "AD"
        assert config.get_label_name(99) == "Unknown(99)"

    def test_get_subject_id_space_format(self):
        """Test subject ID extraction from space-separated format."""
        config = GreekRestingConfig()
        path = Path("/data/HC-RAW/FILT/i002 20150109 1027/i002_20150109_1027_eeg.fif")
        assert config.get_subject_id(path) == "i002"

    def test_get_subject_id_underscore_format(self):
        """Test subject ID extraction from underscore-separated format."""
        config = GreekRestingConfig()
        path = Path("/data/MCI-RAW/FILT/I040_20150702_1203/I040_20150702_1203_eeg.fif")
        assert config.get_subject_id(path) == "I040"


class TestMeditationBIDSConfig:
    """Tests for MeditationBIDSConfig."""

    def test_defaults(self):
        """Test default meditation BIDS configuration."""
        config = MeditationBIDSConfig()
        assert config.name == "meditation_bids"
        assert config.paradigm == "resting"  # Treated as resting with probes
        assert len(config.groups) == 2  # expert, novice
        assert config.file_pattern == "*_task-meditation_eeg.bdf"
        assert config.participants_file == "participants.tsv"

    def test_get_subject_id_from_path(self):
        """Test subject ID extraction from BIDS path."""
        config = MeditationBIDSConfig()

        # From directory structure
        path = Path("/data/sub-001/ses-01/eeg/sub-001_ses-01_task-meditation_eeg.bdf")
        assert config.get_subject_id(path) == "sub-001"

        # From filename fallback
        path = Path("/flat/sub-002_ses-01_task-meditation_eeg.bdf")
        assert config.get_subject_id(path) == "sub-002"


class TestDatasetRegistry:
    """Tests for dataset registry functions."""

    def test_list_datasets(self):
        """Test listing registered datasets."""
        datasets = list_datasets()
        assert "greek_resting" in datasets
        assert "meditation_bids" in datasets

    def test_get_dataset_config(self):
        """Test getting dataset config by name."""
        config = get_dataset_config("greek_resting")
        assert isinstance(config, GreekRestingConfig)

        config = get_dataset_config("meditation_bids")
        assert isinstance(config, MeditationBIDSConfig)

    def test_get_unknown_dataset_raises(self):
        """Test that unknown dataset name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset_config("nonexistent_dataset")

    def test_register_dataset_decorator(self):
        """Test registering a custom dataset."""
        from dataclasses import dataclass

        @register_dataset("test_dataset")
        @dataclass
        class TestDatasetConfig(DatasetConfig):
            name: str = "test_dataset"

            def get_subject_id(self, file_path: Path) -> str:
                return "test_subject"

            def get_files_for_group(self, data_dir: Path, group: GroupConfig) -> list[Path]:
                return []

            def load_raw(self, file_path: Path):
                return None

        assert "test_dataset" in DATASET_REGISTRY
        config = get_dataset_config("test_dataset")
        assert config.name == "test_dataset"

        # Clean up
        del DATASET_REGISTRY["test_dataset"]


class TestDatasetFromHydraConfig:
    """Tests for creating DatasetConfig from Hydra config."""

    def test_basic_hydra_config(self):
        """Test creating config from minimal Hydra config."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "data": {
                "dataset": "greek_resting",
                "preprocessing": {
                    "filter_low": 2.0,
                    "filter_high": 100.0,
                },
            }
        })

        config = dataset_from_hydra_config(cfg)
        assert isinstance(config, GreekRestingConfig)
        assert config.preprocessing.filter_low == 2.0
        assert config.preprocessing.filter_high == 100.0

    def test_default_dataset_name(self):
        """Test that greek_resting is the default dataset."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "data": {
                "preprocessing": {
                    "filter_low": 1.0,
                    "filter_high": 48.0,
                },
            }
        })

        config = dataset_from_hydra_config(cfg)
        assert isinstance(config, GreekRestingConfig)

    def test_groups_override(self):
        """Test overriding groups from Hydra config."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "data": {
                "dataset": "greek_resting",
                "groups": [
                    {"name": "HC", "label": 0, "path": "HC-RAW/FILT"},
                    {"name": "MCI", "label": 1, "path": "MCI-RAW/FILT"},
                    # Only 2 groups, not 3
                ],
                "preprocessing": {
                    "filter_low": 1.0,
                    "filter_high": 48.0,
                },
            }
        })

        config = dataset_from_hydra_config(cfg)
        assert len(config.groups) == 2
        assert config.groups[0].name == "HC"
        assert config.groups[1].name == "MCI"


class TestBIDSIntegration:
    """Integration tests for BIDS dataset support."""

    def test_participants_tsv_parsing(self):
        """Test parsing participants.tsv file."""
        config = MeditationBIDSConfig()

        # Create a temporary BIDS-like structure
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # Create participants.tsv
            participants_tsv = data_dir / "participants.tsv"
            participants_tsv.write_text(
                "participant_id\tgender\tage\tgroup\n"
                "sub-001\tM\t32\texpert\n"
                "sub-002\tF\t35\tnovice\n"
                "sub-003\tM\t28\texpert\n"
            )

            # Create subject directories with BDF files
            for subj in ["sub-001", "sub-002", "sub-003"]:
                subj_dir = data_dir / subj / "ses-01" / "eeg"
                subj_dir.mkdir(parents=True)
                # Create empty BDF file (we won't actually load it)
                bdf_file = subj_dir / f"{subj}_ses-01_task-meditation_eeg.bdf"
                bdf_file.touch()

            # Test file discovery for expert group
            expert_group = GroupConfig(name="expert", label=0, path=".")
            expert_files = config.get_files_for_group(data_dir, expert_group)

            # Should find 2 expert files (sub-001, sub-003)
            assert len(expert_files) == 2
            expert_subjects = {config.get_subject_id(f) for f in expert_files}
            assert expert_subjects == {"sub-001", "sub-003"}

            # Test file discovery for novice group
            novice_group = GroupConfig(name="novice", label=1, path=".")
            novice_files = config.get_files_for_group(data_dir, novice_group)

            # Should find 1 novice file (sub-002)
            assert len(novice_files) == 1
            assert config.get_subject_id(novice_files[0]) == "sub-002"

    def test_validation(self):
        """Test BIDS dataset validation."""
        config = MeditationBIDSConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # Missing participants.tsv should fail
            assert not config.validate(data_dir)

            # Add participants.tsv but no subject folders
            (data_dir / "participants.tsv").write_text(
                "participant_id\tgroup\nsub-001\texpert\n"
            )
            assert not config.validate(data_dir)

            # Add a subject folder
            (data_dir / "sub-001").mkdir()
            assert config.validate(data_dir)
