"""
Dataset Configuration Protocol and Base Classes

Provides a unified interface for different EEG datasets, supporting:
- Resting-state paradigms (continuous recording, chunk-based)
- Task-based paradigms (event-locked epochs)
- Different file formats (FIF, BDF, BIDS)
- Configurable group/label mappings
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Any, Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class GroupConfig:
    """Configuration for a single experimental group/condition."""

    name: str           # Display name (e.g., "HC", "Expert")
    label: int          # Numeric label for classification (0, 1, 2, ...)
    path: str           # Relative path from data_dir to group folder

    def __post_init__(self):
        if self.label < 0:
            raise ValueError(f"Label must be non-negative, got {self.label}")


@dataclass
class PreprocessingConfig:
    """Preprocessing parameters for EEG data."""

    filter_low: float = 1.0        # High-pass cutoff (Hz)
    filter_high: float = 48.0      # Low-pass cutoff (Hz)
    notch_freq: float | None = None  # Line noise frequency (50 or 60 Hz)
    reference: str = "average"     # Reference method: "average", "csd", "mastoids"
    resample_freq: float | None = None  # Target sampling rate (None = keep original)

    # Chunking (for resting-state)
    chunk_duration: float = 5.0    # Chunk length in seconds
    chunk_overlap: float = 0.0     # Overlap between chunks in seconds

    # Epoching (for task-based)
    epoch_tmin: float = -0.2       # Epoch start relative to event (seconds)
    epoch_tmax: float = 0.8        # Epoch end relative to event (seconds)
    baseline: tuple[float, float] | None = (-0.2, 0)  # Baseline correction window


@dataclass
class DatasetConfig(ABC):
    """
    Abstract base class for dataset configurations.

    Each dataset type should subclass this and implement the abstract methods.
    The config defines how to discover, load, and preprocess data.
    """

    # Dataset metadata
    name: str = ""
    description: str = ""
    paradigm: Literal["resting", "task"] = "resting"

    # Groups/conditions
    groups: list[GroupConfig] = field(default_factory=list)

    # File discovery
    file_pattern: str = "*_eeg.fif"  # Glob pattern for data files
    exclude_patterns: list[str] = field(default_factory=list)  # Patterns to skip

    # Preprocessing defaults
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    # Channel selection
    channel_types: list[str] = field(default_factory=lambda: ["eeg"])
    exclude_channels: list[str] = field(default_factory=list)

    @abstractmethod
    def get_subject_id(self, file_path: Path) -> str:
        """
        Extract subject ID from a file path.

        Args:
            file_path: Path to a data file

        Returns:
            Subject identifier string
        """
        pass

    @abstractmethod
    def get_files_for_group(self, data_dir: Path, group: GroupConfig) -> list[Path]:
        """
        Get all valid data files for a group.

        Args:
            data_dir: Root data directory
            group: Group configuration

        Returns:
            List of paths to data files
        """
        pass

    @abstractmethod
    def load_raw(self, file_path: Path) -> Any:
        """
        Load raw data from a file.

        Args:
            file_path: Path to data file

        Returns:
            MNE Raw object (or similar)
        """
        pass

    def get_label_map(self) -> dict[str, int]:
        """Get mapping from group names to numeric labels."""
        return {g.name: g.label for g in self.groups}

    def get_label_name(self, label: int) -> str:
        """Get group name for a numeric label."""
        for g in self.groups:
            if g.label == label:
                return g.name
        return f"Unknown({label})"

    def validate(self, data_dir: Path) -> bool:
        """
        Validate that the dataset exists and is accessible.

        Args:
            data_dir: Root data directory

        Returns:
            True if validation passes
        """
        if not data_dir.exists():
            logger.error(f"Data directory does not exist: {data_dir}")
            return False

        for group in self.groups:
            group_path = data_dir / group.path
            if not group_path.exists():
                logger.warning(f"Group path does not exist: {group_path}")

        return True


# =============================================================================
# CONCRETE IMPLEMENTATIONS
# =============================================================================

@dataclass
class GreekRestingConfig(DatasetConfig):
    """
    Configuration for the Greek HD-EEG resting-state dataset.

    Dataset structure:
        data_dir/
        ├── HC-RAW/FILT/
        │   └── {subject_folder}/
        │       └── *_eeg.fif
        ├── MCI-RAW/FILT/
        │   └── ...
        └── AD-RAW/FILT/
            └── ...

    File naming: {subject_id} {date} {time}.fil/{subject_id}_{date}_{time}_good_1_eeg.fif
    """

    name: str = "greek_resting"
    description: str = "Greek HD-EEG resting-state dataset (HC/MCI/AD)"
    paradigm: Literal["resting", "task"] = "resting"

    file_pattern: str = "*_eeg.fif"
    exclude_patterns: list[str] = field(default_factory=lambda: [
        "-inv.fif",      # Inverse solution
        "-v-inv.fif",    # Vector inverse
        "-ave.fif",      # Averaged
        "-cov.fif",      # Covariance
        "-fwd.fif",      # Forward solution
        "-src.fif",      # Source space
        "-trans.fif",    # Transformation
    ])

    groups: list[GroupConfig] = field(default_factory=lambda: [
        GroupConfig("HC", 0, "HC-RAW/FILT"),
        GroupConfig("MCI", 1, "MCI-RAW/FILT"),
        GroupConfig("AD", 2, "AD-RAW/FILT"),
    ])

    preprocessing: PreprocessingConfig = field(default_factory=lambda: PreprocessingConfig(
        filter_low=1.0,
        filter_high=120.0,
        reference="csd",
        chunk_duration=5.0,
    ))

    def get_subject_id(self, file_path: Path) -> str:
        """Extract subject ID from Greek dataset path."""
        folder_name = file_path.parent.name
        # Handle both "i002 20150109 1027" and "I040_20150702_1203" formats
        if " " in folder_name:
            return folder_name.split()[0]
        elif "_" in folder_name:
            return folder_name.split("_")[0]
        return folder_name

    def get_files_for_group(self, data_dir: Path, group: GroupConfig) -> list[Path]:
        """Get all FIF files for a group."""
        group_path = data_dir / group.path
        if not group_path.exists():
            logger.warning(f"Group path does not exist: {group_path}")
            return []

        files = []
        for fif_file in sorted(group_path.rglob(self.file_pattern)):
            # Skip excluded patterns
            filename = fif_file.name.lower()
            if any(pat in filename for pat in self.exclude_patterns):
                continue
            # Skip macOS resource forks
            if fif_file.name.startswith("._"):
                continue
            files.append(fif_file)

        return files

    def load_raw(self, file_path: Path) -> Any:
        """Load MNE Raw from FIF file."""
        import mne
        mne.set_log_level("WARNING")
        return mne.io.read_raw_fif(file_path, preload=True)


@dataclass
class MeditationBIDSConfig(DatasetConfig):
    """
    Configuration for the OpenNeuro Meditation/Mind-wandering dataset (ds001787).

    BIDS structure:
        data_dir/
        ├── participants.tsv
        ├── sub-001/
        │   ├── ses-01/eeg/
        │   │   ├── sub-001_ses-01_task-meditation_eeg.bdf
        │   │   └── sub-001_ses-01_task-meditation_events.tsv
        │   └── ses-02/eeg/...
        └── sub-024/...

    Two-class classification: Expert vs Novice meditators
    """

    name: str = "meditation_bids"
    description: str = "OpenNeuro meditation study - expert vs novice (ds001787)"
    paradigm: Literal["resting", "task"] = "resting"  # Treated as resting with probes

    file_pattern: str = "*_task-meditation_eeg.bdf"
    exclude_patterns: list[str] = field(default_factory=list)

    # Groups are determined from participants.tsv
    groups: list[GroupConfig] = field(default_factory=lambda: [
        GroupConfig("expert", 0, "."),  # Path is root, subjects determined by participants.tsv
        GroupConfig("novice", 1, "."),
    ])

    preprocessing: PreprocessingConfig = field(default_factory=lambda: PreprocessingConfig(
        filter_low=2.0,       # Paper used 2 Hz high-pass
        filter_high=48.0,
        reference="average",
        chunk_duration=10.0,  # Paper used 10s epochs before probes
        notch_freq=50.0,      # EU line frequency
    ))

    # BIDS-specific
    participants_file: str = "participants.tsv"
    group_column: str = "group"  # Column in participants.tsv with group labels

    # Channel selection - only EEG channels (skip GSR, respiration, etc.)
    channel_types: list[str] = field(default_factory=lambda: ["eeg"])
    n_eeg_channels: int = 64

    def __post_init__(self):
        """Load group assignments from participants.tsv if available."""
        # Groups will be populated when validate() is called with data_dir
        pass

    def get_subject_id(self, file_path: Path) -> str:
        """Extract subject ID from BIDS path (e.g., sub-001)."""
        # Walk up parent directories to find sub-XXX folder
        for parent in file_path.parents:
            if parent.name.startswith("sub-"):
                return parent.name
        # Fallback: extract from filename like sub-001_ses-01_task-meditation_eeg.bdf
        name = file_path.stem
        if "sub-" in name:
            # Extract sub-XXX from filename
            parts = name.split("_")
            for part in parts:
                if part.startswith("sub-"):
                    return part
        return file_path.stem

    def get_files_for_group(self, data_dir: Path, group: GroupConfig) -> list[Path]:
        """
        Get all BDF files for a group based on participants.tsv.

        For BIDS datasets, we read the participants.tsv to determine
        which subjects belong to which group.
        """
        # Read participants file
        participants_path = data_dir / self.participants_file
        if not participants_path.exists():
            logger.error(f"Participants file not found: {participants_path}")
            return []

        # Parse TSV
        import csv
        subject_groups = {}
        with open(participants_path) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                subj_id = row.get("participant_id", "")
                subj_group = row.get(self.group_column, "")
                subject_groups[subj_id] = subj_group

        # Find files for subjects in this group
        files = []
        for subj_id, subj_group in subject_groups.items():
            if subj_group != group.name:
                continue

            # Find all session files for this subject
            subj_dir = data_dir / subj_id
            if not subj_dir.exists():
                continue

            for bdf_file in sorted(subj_dir.rglob(self.file_pattern)):
                if bdf_file.name.startswith("._"):
                    continue
                # Check if file is a symlink to git-annex (not downloaded)
                if bdf_file.is_symlink():
                    target = bdf_file.resolve()
                    if not target.exists():
                        logger.warning(f"Git-annex file not downloaded: {bdf_file}")
                        continue
                files.append(bdf_file)

        return files

    def load_raw(self, file_path: Path) -> Any:
        """Load MNE Raw from BDF file."""
        import mne
        mne.set_log_level("WARNING")

        raw = mne.io.read_raw_bdf(file_path, preload=True)

        # Select only EEG channels
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        if len(eeg_picks) > 0:
            raw = raw.pick(eeg_picks)

        return raw

    def validate(self, data_dir: Path) -> bool:
        """Validate BIDS dataset structure."""
        if not super().validate(data_dir):
            return False

        # Check for participants.tsv
        participants_path = data_dir / self.participants_file
        if not participants_path.exists():
            logger.error(f"BIDS dataset missing participants.tsv: {participants_path}")
            return False

        # Check for at least one subject folder
        subject_dirs = list(data_dir.glob("sub-*"))
        if not subject_dirs:
            logger.error(f"No subject folders (sub-*) found in {data_dir}")
            return False

        logger.info(f"Found {len(subject_dirs)} subject folders")
        return True


# =============================================================================
# DATASET REGISTRY
# =============================================================================

# Registry of available dataset configurations
DATASET_REGISTRY: dict[str, type[DatasetConfig]] = {
    "greek_resting": GreekRestingConfig,
    "meditation_bids": MeditationBIDSConfig,
}


def register_dataset(name: str) -> Callable[[type[DatasetConfig]], type[DatasetConfig]]:
    """
    Decorator to register a dataset configuration.

    Usage:
        @register_dataset("my_dataset")
        class MyDatasetConfig(DatasetConfig):
            ...
    """
    def decorator(cls: type[DatasetConfig]) -> type[DatasetConfig]:
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator


def get_dataset_config(name: str, **overrides) -> DatasetConfig:
    """
    Get a dataset configuration by name.

    Args:
        name: Dataset name (e.g., "greek_resting", "meditation_bids")
        **overrides: Override any config fields

    Returns:
        Instantiated DatasetConfig

    Raises:
        ValueError: If dataset name is not registered
    """
    if name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(f"Unknown dataset: '{name}'. Available: {available}")

    config_cls = DATASET_REGISTRY[name]

    # Create instance with overrides
    if overrides:
        return config_cls(**overrides)
    return config_cls()


def list_datasets() -> list[str]:
    """List all registered dataset names."""
    return sorted(DATASET_REGISTRY.keys())


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_groups_from_dict(groups_dict: list[dict]) -> list[GroupConfig]:
    """
    Create GroupConfig list from Hydra config dict format.

    Args:
        groups_dict: List of dicts with name, label, path keys

    Returns:
        List of GroupConfig objects
    """
    return [GroupConfig(**g) for g in groups_dict]


def dataset_from_hydra_config(cfg) -> DatasetConfig:
    """
    Create DatasetConfig from Hydra configuration.

    Args:
        cfg: Hydra DictConfig with data.dataset and optional overrides

    Returns:
        DatasetConfig instance
    """
    dataset_name = cfg.data.get("dataset", "greek_resting")
    config = get_dataset_config(dataset_name)

    # Override groups if specified
    if "groups" in cfg.data and cfg.data.groups:
        config.groups = create_groups_from_dict(cfg.data.groups)

    # Override preprocessing if specified
    if "preprocessing" in cfg.data:
        prep_cfg = cfg.data.preprocessing
        config.preprocessing = PreprocessingConfig(
            filter_low=prep_cfg.get("filter_low", config.preprocessing.filter_low),
            filter_high=prep_cfg.get("filter_high", config.preprocessing.filter_high),
            notch_freq=prep_cfg.get("notch_freq", config.preprocessing.notch_freq),
            reference=prep_cfg.get("reference", config.preprocessing.reference),
            resample_freq=prep_cfg.get("resample_freq", config.preprocessing.resample_freq),
            chunk_duration=prep_cfg.get("chunk_duration", config.preprocessing.chunk_duration),
            chunk_overlap=prep_cfg.get("chunk_overlap", config.preprocessing.chunk_overlap),
        )

    return config
