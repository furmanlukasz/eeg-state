"""
Environment-aware path resolution for local Mac and RunPod GPU.

Detects environment automatically via hostname/directory presence,
or accepts explicit override via EEG_DATA_ROOT environment variable.

Usage:
    from benchmarks.paths import get_paths

    paths = get_paths()
    print(paths.data_root)        # /Volumes/Nvme_Data or /workspace/data
    print(paths.greek_data)       # .../GreekData or .../data
    print(paths.meditation_data)  # .../ds001787
    print(paths.checkpoints)      # dict of dataset -> checkpoint path
    print(paths.output_root)      # .../results/benchmarks
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class BenchmarkPaths:
    """Resolved paths for the current environment."""

    environment: str  # "local_mac", "runpod", or "custom"
    data_root: Path
    repo_root: Path
    output_root: Path

    # Dataset-specific data directories
    greek_data: Optional[Path] = None
    meditation_data: Optional[Path] = None
    electronic_data: Optional[Path] = None

    # Checkpoint paths (dataset name -> path)
    checkpoints: dict[str, Path] = field(default_factory=dict)

    # Device for inference
    device: str = "cpu"

    def validate(self) -> list[str]:
        """Check which datasets are available. Returns list of warnings."""
        warnings = []
        if self.greek_data and not self.greek_data.exists():
            warnings.append(f"Greek data not found: {self.greek_data}")
        if self.meditation_data and not self.meditation_data.exists():
            warnings.append(f"Meditation data not found: {self.meditation_data}")
        if self.electronic_data and not self.electronic_data.exists():
            warnings.append(f"Electronic oscillator data not found: {self.electronic_data}")
        for name, cp in self.checkpoints.items():
            if not cp.exists():
                warnings.append(f"Checkpoint not found for {name}: {cp}")
        return warnings

    def available_datasets(self) -> list[str]:
        """Return list of datasets with existing data directories."""
        available = []
        if self.greek_data and self.greek_data.exists():
            available.append("greek_resting")
        if self.meditation_data and self.meditation_data.exists():
            available.append("meditation_bids")
        if self.electronic_data and self.electronic_data.exists():
            available.append("electronic_oscillators")
        return available


def detect_environment() -> str:
    """Detect current execution environment."""
    # Explicit override
    if os.environ.get("EEG_ENVIRONMENT"):
        return os.environ["EEG_ENVIRONMENT"]

    # RunPod detection: /workspace exists and we're on Linux
    if Path("/workspace/data").exists() and platform.system() == "Linux":
        return "runpod"

    # Mac detection
    if platform.system() == "Darwin":
        return "local_mac"

    return "unknown"


def get_paths(
    environment: Optional[str] = None,
    data_root: Optional[str] = None,
) -> BenchmarkPaths:
    """
    Get resolved paths for the current environment.

    Args:
        environment: Override auto-detection ("local_mac", "runpod", "custom")
        data_root: Override data root directory (also settable via EEG_DATA_ROOT)

    Returns:
        BenchmarkPaths with all paths resolved for the current environment
    """
    env = environment or detect_environment()

    # Repo root: find it relative to this file
    repo_root = Path(__file__).parent.parent.resolve()

    # Data root: explicit > env var > environment default
    if data_root:
        root = Path(data_root)
    elif os.environ.get("EEG_DATA_ROOT"):
        root = Path(os.environ["EEG_DATA_ROOT"])
    elif env == "runpod":
        root = Path("/workspace/data")
    elif env == "local_mac":
        root = Path("/Volumes/Nvme_Data")
    else:
        root = repo_root / "data"

    # Output directory
    output_root = Path(os.environ.get(
        "EEG_OUTPUT_ROOT",
        str(repo_root / "results" / "benchmarks"),
    ))

    # Device
    if env == "runpod":
        device = "cuda"
    elif env == "local_mac":
        device = "mps"
    else:
        device = "cpu"

    # Dataset paths differ by environment
    if env == "runpod":
        # RunPod: Greek data is flat under /workspace/data/{MCI,HID,AD}/FILT/
        # Meditation is under /workspace/data/ds001787/
        greek_data = root  # The root IS the Greek data dir on RunPod
        meditation_data = root / "ds001787"
        electronic_data = root / "electronic_oscillators"  # Future

        checkpoints = {
            "greek_resting": repo_root / "models" / "best.pt",
            "meditation_bids": _find_meditation_checkpoint(repo_root),
        }

    elif env == "local_mac":
        # Local Mac: datasets in separate dirs under /Volumes/Nvme_Data
        greek_data = root / "GreekData"
        meditation_data = root / "ds001787"
        electronic_data = root / "electronic_oscillators"  # Future

        checkpoints = {
            "greek_resting": repo_root / "models" / "best_MCI_AD_HC.pt",
            "meditation_bids": _find_meditation_checkpoint(repo_root),
        }

    else:
        # Generic: everything under data_root
        greek_data = root
        meditation_data = root / "ds001787"
        electronic_data = root / "electronic_oscillators"
        checkpoints = {}

    return BenchmarkPaths(
        environment=env,
        data_root=root,
        repo_root=repo_root,
        output_root=output_root,
        greek_data=greek_data,
        meditation_data=meditation_data,
        electronic_data=electronic_data,
        checkpoints=checkpoints,
        device=device,
    )


def _find_meditation_checkpoint(repo_root: Path) -> Path:
    """Find the best meditation model checkpoint."""
    # Try known locations
    candidates = [
        repo_root / "outputs" / "transformer_ff384_meditation_64ch_noavg" / "meditation_bids",
        repo_root / "models" / "best_meditation.pt",
    ]

    for candidate in candidates:
        if candidate.is_dir():
            # Find timestamped subdirectory with best.pt
            for subdir in sorted(candidate.iterdir(), reverse=True):
                best = subdir / "checkpoints" / "best.pt"
                if best.exists():
                    return best
        elif candidate.exists():
            return candidate

    # Fallback
    return repo_root / "models" / "best_meditation.pt"
