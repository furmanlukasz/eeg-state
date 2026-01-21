"""
Local Analysis Configuration

Edit these paths to match your local setup.
"""

from pathlib import Path

# =============================================================================
# PATHS - Edit these to match your local setup
# =============================================================================

# Model checkpoint (from MatrixAutoEncoder repo or RunPod)
CHECKPOINT_PATH = Path("/Users/luki/Documents/GitHub/eeg-state-biomarkers/models/best2.pt")

# Data directories - point to your local .fif files
# Full dataset on mounted NVMe drive
DATA_DIR = Path("/Volumes/Nvme_Data/GreekData")

# Subdirectory mapping for different conditions
# AD -> AD-RAW/FILT, MCI -> MCI-RAW/FILT, HC/HID -> HC-RAW/FILT
CONDITION_SUBDIRS = {
    "AD": "AD-RAW/FILT",
    "MCI": "MCI-RAW/FILT",
    "HID": "HC-RAW/FILT",
    "HC": "HC-RAW/FILT",
}

# Output directory for plots
OUTPUT_DIR = Path("/Users/luki/Documents/GitHub/eeg-state-biomarkers/results/local_analysis")

# =============================================================================
# ANALYSIS PARAMETERS
# =============================================================================

# Sampling frequency of EEG data
SFREQ = 250.0

# Filter settings for phase extraction
# Default: 1-120Hz (broadband including high gamma)
# Note: High frequencies (>40Hz) may contain EMG artifacts in frontal/temporal channels
FILTER_LOW = 1.0
FILTER_HIGH = 120.0

# Chunk duration in seconds
CHUNK_DURATION = 5.0

# RQA parameters
RR_TARGETS = [0.01, 0.02, 0.05]  # 1%, 2%, 5%
THEILER_WINDOW = 50  # samples (~0.2 seconds)

# UMAP parameters
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_N_COMPONENTS = 3  # 3D for visualization

# Device for model inference
DEVICE = "mps"  # Use "mps" for M1 Mac, "cuda" for GPU, "cpu" for CPU

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def get_fif_files(conditions=None):
    """
    Get list of .fif files from data directory.

    Args:
        conditions: List of conditions to include (e.g., ["AD", "MCI", "HID"])
                   If None, includes all.

    Returns:
        List of (fif_path, label, condition) tuples
        Labels: HC/HID=0, MCI=1, AD=2
    """
    if conditions is None:
        conditions = ["AD", "MCI", "HID"]

    # Patterns to exclude (inverse solutions, processed files, etc.)
    # These are source-space reconstructions, not raw EEG
    exclude_patterns = [
        "-inv.fif",      # Inverse solution
        "-v-inv.fif",    # Vector inverse solution
        "-ave.fif",      # Averaged evoked
        "-cov.fif",      # Covariance matrix
        "-fwd.fif",      # Forward solution
        "-src.fif",      # Source space
        "-trans.fif",    # Transformation matrix
    ]

    # Label mapping: HC/HID=0 (healthy), MCI=1, AD=2
    label_map = {"HC": 0, "HID": 0, "MCI": 1, "AD": 2}

    files = []
    for condition in conditions:
        # Use condition-specific subdirectory
        subdir = CONDITION_SUBDIRS.get(condition, condition)
        condition_dir = DATA_DIR / subdir

        if condition_dir.exists():
            label = label_map.get(condition, 1)  # Default to 1 if unknown
            for fif_file in sorted(condition_dir.glob("**/*.fif")):
                # Skip non-raw files (inverse solutions, etc.)
                filename = fif_file.name.lower()
                if any(pattern in filename for pattern in exclude_patterns):
                    continue
                # Only include files with "_eeg.fif" suffix (raw EEG)
                if "_eeg.fif" in filename:
                    files.append((fif_file, label, condition))
        else:
            print(f"Warning: Directory not found: {condition_dir}")

    return files


def get_subject_id(fif_path):
    """
    Extract subject ID from file path.

    Examples:
        .../i002 20150109 1027.fil/i002 20150109 1027.fil_good_1_eeg.fif -> i002
        .../I040_20150702_1203/I040_20150702_1203_good_1_eeg.fif -> I040
        .../S017 20140124 0857.fil/S017 20140124 0857.fil_good_1_eeg.fif -> S017
    """
    # Get the parent folder name (subject folder)
    folder_name = fif_path.parent.name
    # Extract first part (subject ID) - split on space or underscore
    if " " in folder_name:
        return folder_name.split()[0]
    elif "_" in folder_name:
        return folder_name.split("_")[0]
    else:
        return folder_name


def get_unique_subjects(fif_files: list) -> dict:
    """
    Group fif files by unique subject ID.

    Args:
        fif_files: List of (fif_path, label, condition) tuples

    Returns:
        Dict mapping subject_id -> (fif_path, label, condition)
        Only returns first file per subject (avoids duplicates)
    """
    subjects = {}
    for fif_path, label, condition in fif_files:
        subject_id = get_subject_id(fif_path)
        if subject_id not in subjects:
            subjects[subject_id] = (fif_path, label, condition)
    return subjects


def get_subjects_by_group(fif_files: list) -> dict:
    """
    Get unique subjects separated by group.

    Args:
        fif_files: List of (fif_path, label, condition) tuples

    Returns:
        Dict with keys 'hc', 'mci', 'ad', each containing list of
        (fif_path, label, condition, subject_id) tuples
    """
    subjects = get_unique_subjects(fif_files)

    groups = {
        "hc": [],   # label=0 (HC/HID)
        "mci": [],  # label=1
        "ad": [],   # label=2
    }

    for subject_id, (fif_path, label, condition) in subjects.items():
        entry = (fif_path, label, condition, subject_id)
        if label == 0:
            groups["hc"].append(entry)
        elif label == 1:
            groups["mci"].append(entry)
        elif label == 2:
            groups["ad"].append(entry)

    return groups


def get_label_name(label: int) -> str:
    """Get human-readable name for label."""
    return {0: "HC", 1: "MCI", 2: "AD"}.get(label, "Unknown")
