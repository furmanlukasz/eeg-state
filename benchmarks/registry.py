"""
Dataset and Method registry for benchmarking.

Provides a unified gateway for loading any registered dataset with
consistent output format, and a method registry for analysis pipelines.

Usage:
    from benchmarks.registry import load_dataset, list_methods

    # Load dataset (respects --quick mode)
    data = load_dataset("greek_resting", quick=True)
    # Returns: DataBundle(files, labels, groups, subject_ids, config)

    # List available methods
    methods = list_methods()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from benchmarks.paths import BenchmarkPaths, get_paths

logger = logging.getLogger(__name__)


@dataclass
class DataBundle:
    """Standardized output from dataset loading."""

    files: list[Path]            # Paths to data files
    labels: list[int]            # Numeric labels (0, 1, 2, ...)
    group_names: list[str]       # Group name per file ("HC", "MCI", "expert", ...)
    subject_ids: list[str]       # Subject ID per file
    dataset_name: str            # Registry name
    n_groups: int = 0            # Number of unique groups
    group_info: dict = field(default_factory=dict)  # Group name -> count

    def __post_init__(self):
        self.n_groups = len(set(self.group_names))
        from collections import Counter
        self.group_info = dict(Counter(self.group_names))

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [f"Dataset: {self.dataset_name}"]
        lines.append(f"  Files: {len(self.files)}")
        lines.append(f"  Subjects: {len(set(self.subject_ids))}")
        lines.append(f"  Groups: {self.n_groups}")
        for name, count in sorted(self.group_info.items()):
            lines.append(f"    {name}: {count} files")
        return "\n".join(lines)


@dataclass
class BenchmarkResult:
    """Standardized output from a benchmark method."""

    dataset_name: str
    method_name: str
    features: Any              # numpy array (n_samples, n_features) or dict
    labels: list[int]          # Labels aligned with features
    subject_ids: list[str]     # Subject IDs aligned with features
    metadata: dict = field(default_factory=dict)  # Method-specific info


def load_dataset(
    dataset_name: str,
    paths: Optional[BenchmarkPaths] = None,
    quick: bool = False,
    max_subjects_per_group: Optional[int] = None,
) -> DataBundle:
    """
    Load a dataset using the DatasetConfig system.

    Args:
        dataset_name: Registered dataset name ("greek_resting", "meditation_bids")
        paths: Pre-resolved paths (auto-detected if None)
        quick: If True, limit to 2-3 subjects per group for fast iteration
        max_subjects_per_group: Override max subjects (quick sets this to 3)

    Returns:
        DataBundle with files, labels, group_names, subject_ids
    """
    import sys
    # Ensure eeg_biomarkers is importable
    if paths is None:
        paths = get_paths()
    sys.path.insert(0, str(paths.repo_root / "src"))

    from eeg_biomarkers.data.dataset_config import get_dataset_config

    config = get_dataset_config(dataset_name)

    # Determine data directory
    data_dir_map = {
        "greek_resting": paths.greek_data,
        "meditation_bids": paths.meditation_data,
        "electronic_oscillators": paths.electronic_data,
    }
    data_dir = data_dir_map.get(dataset_name)
    if data_dir is None or not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory for '{dataset_name}' not found: {data_dir}\n"
            f"Environment: {paths.environment}, data_root: {paths.data_root}"
        )

    # Quick mode: limit subjects
    n_per_group = max_subjects_per_group or (3 if quick else None)

    files = []
    labels = []
    group_names = []
    subject_ids = []

    for group in config.groups:
        group_files = config.get_files_for_group(data_dir, group)

        if not group_files:
            logger.warning(f"No files found for group '{group.name}' in {data_dir}")
            continue

        # Deduplicate by subject (take first file per subject)
        seen_subjects = set()
        unique_files = []
        for f in group_files:
            sid = config.get_subject_id(f)
            if sid not in seen_subjects:
                seen_subjects.add(sid)
                unique_files.append((f, sid))

        # Limit subjects if quick mode
        if n_per_group and len(unique_files) > n_per_group:
            unique_files = unique_files[:n_per_group]
            logger.info(
                f"Quick mode: limited {group.name} to {n_per_group} subjects "
                f"(from {len(seen_subjects)})"
            )

        for f, sid in unique_files:
            files.append(f)
            labels.append(group.label)
            group_names.append(group.name)
            subject_ids.append(sid)

    logger.info(
        f"Loaded {dataset_name}: {len(files)} files, "
        f"{len(set(subject_ids))} subjects, "
        f"{len(set(group_names))} groups"
    )

    return DataBundle(
        files=files,
        labels=labels,
        group_names=group_names,
        subject_ids=subject_ids,
        dataset_name=dataset_name,
    )


# =============================================================================
# METHOD REGISTRY
# =============================================================================

# Method functions: (DataBundle, BenchmarkPaths, **kwargs) -> BenchmarkResult
METHOD_REGISTRY: dict[str, Callable] = {}


def register_method(name: str):
    """Decorator to register a benchmark method."""
    def decorator(func: Callable) -> Callable:
        METHOD_REGISTRY[name] = func
        return func
    return decorator


def list_methods() -> list[str]:
    """List all registered method names."""
    return sorted(METHOD_REGISTRY.keys())


def run_method(
    method_name: str,
    data: DataBundle,
    paths: Optional[BenchmarkPaths] = None,
    **kwargs,
) -> BenchmarkResult:
    """
    Run a registered method on a dataset.

    Args:
        method_name: Registered method name
        data: DataBundle from load_dataset()
        paths: BenchmarkPaths for checkpoint/output resolution
        **kwargs: Method-specific arguments

    Returns:
        BenchmarkResult
    """
    if method_name not in METHOD_REGISTRY:
        available = ", ".join(list_methods())
        raise ValueError(f"Unknown method: '{method_name}'. Available: {available}")

    if paths is None:
        paths = get_paths()

    method_fn = METHOD_REGISTRY[method_name]
    return method_fn(data, paths, **kwargs)
