#!/usr/bin/env python3
"""
Simple RQA-based Classification (No CV)

Train/test split classification using RQA features from autoencoder latent space.
Generates ROC curves and feature importance plots.

Usage:
    python classify_rqa.py                           # Default: HC vs MCI
    python classify_rqa.py --conditions HID MCI      # HC vs MCI
    python classify_rqa.py --conditions HID AD       # HC vs AD
    python classify_rqa.py --conditions HID MCI AD   # All three (binary: HC vs impaired)
    python classify_rqa.py --n-subjects 10           # Use 10 subjects per group
    python classify_rqa.py --n-chunks 5              # Use 5 chunks per subject
    python classify_rqa.py --rr-target 0.05          # 5% recurrence rate
    python classify_rqa.py --test-size 0.3           # 30% test split
    python classify_rqa.py --random-weights          # Use random weights (baseline)
    python classify_rqa.py --list-subjects           # List available subjects
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix,
)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CHECKPOINT_PATH, DATA_DIR, OUTPUT_DIR, DEVICE,
    FILTER_LOW, FILTER_HIGH, CHUNK_DURATION, SFREQ,
    THEILER_WINDOW, ensure_output_dir, get_fif_files,
    get_subjects_by_group, get_label_name
)
from load_model import load_model_from_checkpoint, create_model, compute_latent_trajectory
from load_data import load_and_preprocess_fif
from plot_recurrence import compute_angular_distance_matrix, compute_recurrence_matrix, compute_rqa_stats

# Check for XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import RandomForestClassifier

# Color mapping
GROUP_COLORS = {0: "blue", 1: "orange", 2: "red"}
GROUP_NAMES = {0: "HC", 1: "MCI", 2: "AD"}


def extract_rqa_features(
    model,
    fif_path: Path,
    model_info: dict,
    rr_target: float = 0.02,
    theiler: int = 50,
    n_chunks: int = 1,
) -> list[dict]:
    """
    Extract RQA features from all chunks of a subject.

    Returns:
        List of dicts with RQA features for each chunk
    """
    data = load_and_preprocess_fif(
        fif_path, FILTER_LOW, FILTER_HIGH, CHUNK_DURATION,
        include_amplitude=model_info["include_amplitude"],
        verbose=False,
    )

    available_chunks = len(data["chunks"])
    if available_chunks == 0:
        return []

    chunks_to_use = list(range(min(n_chunks, available_chunks)))

    features_list = []
    for cidx in chunks_to_use:
        phase_data = data["chunks"][cidx]
        latent = compute_latent_trajectory(model, phase_data, DEVICE)
        distance_matrix = compute_angular_distance_matrix(latent)
        R, eps = compute_recurrence_matrix(distance_matrix, rr_target, theiler)
        stats = compute_rqa_stats(R)
        features_list.append(stats)

    return features_list


def collect_features_for_groups(
    model,
    model_info: dict,
    groups: dict,
    n_subjects_per_group: int,
    n_chunks_per_subject: int,
    rr_target: float,
    theiler: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Collect RQA features from all groups.

    Returns:
        features: (n_samples, n_features) array
        labels: (n_samples,) array with 0=HC, 1=impaired
        subject_ids: list of subject IDs
        feature_names: list of feature names
    """
    all_features = []
    all_labels = []
    all_subject_ids = []
    feature_names = None

    for group_key, subjects in groups.items():
        if not subjects:
            continue

        group_name = GROUP_NAMES.get(subjects[0][1], group_key.upper())
        # Binary classification: HC (label=0) vs impaired (MCI/AD, label=1)
        binary_label = 0 if subjects[0][1] == 0 else 1

        print(f"\nExtracting features from {group_name} (targeting {n_subjects_per_group} subjects)...")

        subjects_processed = 0
        for fif_path, label, condition, subject_id in subjects:
            if subjects_processed >= n_subjects_per_group:
                break

            print(f"  {subject_id} ({condition})...", end=" ")

            features_list = extract_rqa_features(
                model, fif_path, model_info,
                rr_target=rr_target,
                theiler=theiler,
                n_chunks=n_chunks_per_subject,
            )

            if not features_list:
                print("no chunks, skipping")
                continue

            print(f"{len(features_list)} chunks")

            # Store feature names from first result
            if feature_names is None:
                feature_names = list(features_list[0].keys())

            # Add features for each chunk
            for feat_dict in features_list:
                feat_vector = [feat_dict[k] for k in feature_names]
                all_features.append(feat_vector)
                all_labels.append(binary_label)
                all_subject_ids.append(subject_id)

            subjects_processed += 1

        if subjects_processed < n_subjects_per_group:
            print(f"  NOTE: Only found {subjects_processed} valid subjects")

    return (
        np.array(all_features),
        np.array(all_labels),
        all_subject_ids,
        feature_names,
    )


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path,
    title_suffix: str = "",
    show_plot: bool = True,
):
    """Plot ROC curve with AUC."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve: HC vs Impaired{title_suffix}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Add AUC annotation
    ax.annotate(f'AUC = {auc:.3f}', xy=(0.6, 0.2), fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    save_path = output_dir / "roc_curve.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return auc


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    output_dir: Path,
    show_plot: bool = True,
):
    """Plot feature importance from classifier."""
    # Sort by importance
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(feature_names)))
    colors = colors[indices][::-1]  # Reverse so most important is darkest

    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, importances[indices][::-1], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('RQA Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    save_path = output_dir / "feature_importance.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    show_plot: bool = True,
):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = ['HC', 'Impaired']
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes, yticklabels=classes,
        ylabel='True label',
        xlabel='Predicted label',
    )
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16, fontweight='bold')

    plt.tight_layout()

    save_path = output_dir / "confusion_matrix.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def list_subjects(groups: dict):
    """List available subjects by group."""
    print("\n" + "=" * 60)
    print("AVAILABLE SUBJECTS")
    print("=" * 60)

    for group_key in ["hc", "mci", "ad"]:
        subjects = groups.get(group_key, [])
        if not subjects:
            continue

        group_name = GROUP_NAMES.get(subjects[0][1], group_key.upper())
        print(f"\n{group_name} subjects ({len(subjects)}):")
        for i, (fif_path, label, condition, subject_id) in enumerate(subjects):
            print(f"  {i+1:3d}. {subject_id:10s} ({condition})")

    total = sum(len(groups.get(k, [])) for k in ["hc", "mci", "ad"])
    print(f"\nTotal: {total} unique subjects")


def main():
    parser = argparse.ArgumentParser(
        description="Simple RQA-based classification (no CV)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python classify_rqa.py                           # Default: HC vs MCI
  python classify_rqa.py --conditions HID MCI      # HC vs MCI
  python classify_rqa.py --conditions HID AD       # HC vs AD
  python classify_rqa.py --conditions HID MCI AD   # HC vs all impaired
  python classify_rqa.py --n-subjects 10           # 10 subjects per group
  python classify_rqa.py --n-chunks 5              # 5 chunks per subject
  python classify_rqa.py --rr-target 0.05          # 5% recurrence rate
  python classify_rqa.py --test-size 0.3           # 30% test split
  python classify_rqa.py --list-subjects           # List available subjects
        """
    )
    parser.add_argument("--n-subjects", type=int, default=10,
                        help="Number of subjects per group (default: 10)")
    parser.add_argument("--n-chunks", type=int, default=3,
                        help="Number of chunks per subject (default: 3)")
    parser.add_argument("--conditions", type=str, nargs="+", default=["HID", "MCI"],
                        help="Conditions to compare: HID, MCI, AD (default: HID MCI)")
    parser.add_argument("--rr-target", type=float, default=0.02,
                        help="Target recurrence rate (default: 0.02 = 2%%)")
    parser.add_argument("--theiler", type=int, default=THEILER_WINDOW,
                        help=f"Theiler window in samples (default: {THEILER_WINDOW})")
    parser.add_argument("--no-theiler", action="store_true",
                        help="Disable Theiler window")
    parser.add_argument("--test-size", type=float, default=0.25,
                        help="Test set proportion (default: 0.25)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--random-weights", action="store_true",
                        help="Use random weights instead of trained model (baseline)")
    parser.add_argument("--list-subjects", action="store_true",
                        help="List available subjects and exit")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display plots interactively")
    args = parser.parse_args()

    # Handle Theiler window
    theiler = 0 if args.no_theiler else args.theiler

    output_dir = ensure_output_dir()

    # Get subjects by group for selected conditions
    fif_files = get_fif_files(args.conditions)
    groups = get_subjects_by_group(fif_files)

    # Print summary
    group_counts = []
    for key in ["hc", "mci", "ad"]:
        if groups.get(key):
            group_counts.append(f"{len(groups[key])} {key.upper()}")
    print(f"Found {', '.join(group_counts)} subjects for conditions: {args.conditions}")

    # Handle --list-subjects
    if args.list_subjects:
        list_subjects(groups)
        return 0

    # Check we have at least two groups
    active_groups = {k: v for k, v in groups.items() if v}
    if len(active_groups) < 2:
        print("Need at least 2 groups with subjects for classification!")
        print("Available groups:", list(active_groups.keys()))
        return 1

    # Check for HC group (required for binary classification)
    if "hc" not in active_groups:
        print("ERROR: Need HC/HID group for binary classification!")
        return 1

    # Load model
    print("\nLoading model...")
    model_info = load_model_from_checkpoint(CHECKPOINT_PATH, DEVICE)

    # Create model
    first_subject = list(active_groups.values())[0][0]
    first_data = load_and_preprocess_fif(
        first_subject[0], FILTER_LOW, FILTER_HIGH, CHUNK_DURATION,
        include_amplitude=model_info["include_amplitude"],
        verbose=False,
    )

    # Use random weights if requested (baseline comparison)
    load_weights = not args.random_weights
    model = create_model(first_data["n_channels"], model_info, DEVICE, load_weights=load_weights)

    if args.random_weights:
        print("\n*** BASELINE MODE: Using RANDOM weights (untrained model) ***")

    # Collect features
    print(f"\nSettings: RR={args.rr_target*100:.0f}%, Theiler={theiler}")
    features, labels, subject_ids, feature_names = collect_features_for_groups(
        model, model_info, active_groups,
        n_subjects_per_group=args.n_subjects,
        n_chunks_per_subject=args.n_chunks,
        rr_target=args.rr_target,
        theiler=theiler,
    )

    print(f"\nCollected {len(features)} samples from {len(set(subject_ids))} subjects")
    print(f"Features: {feature_names}")
    print(f"Class distribution: HC={sum(labels==0)}, Impaired={sum(labels==1)}")

    if len(features) < 10:
        print("ERROR: Not enough samples for classification!")
        return 1

    # Train/test split (stratified by subject to avoid leakage)
    # Group samples by subject first
    unique_subjects = list(set(subject_ids))
    subject_labels = {s: labels[subject_ids.index(s)] for s in unique_subjects}

    # Split subjects
    subjects_0 = [s for s in unique_subjects if subject_labels[s] == 0]
    subjects_1 = [s for s in unique_subjects if subject_labels[s] == 1]

    n_test_0 = max(1, int(len(subjects_0) * args.test_size))
    n_test_1 = max(1, int(len(subjects_1) * args.test_size))

    np.random.seed(args.seed)
    test_subjects_0 = list(np.random.choice(subjects_0, n_test_0, replace=False))
    test_subjects_1 = list(np.random.choice(subjects_1, n_test_1, replace=False))
    test_subjects = set(test_subjects_0 + test_subjects_1)

    # Create train/test masks
    test_mask = np.array([s in test_subjects for s in subject_ids])
    train_mask = ~test_mask

    X_train, X_test = features[train_mask], features[test_mask]
    y_train, y_test = labels[train_mask], labels[test_mask]

    print(f"\nTrain: {len(X_train)} samples ({sum(y_train==0)} HC, {sum(y_train==1)} Impaired)")
    print(f"Test:  {len(X_test)} samples ({sum(y_test==0)} HC, {sum(y_test==1)} Impaired)")
    print(f"Test subjects: {sorted(test_subjects)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train classifier
    print("\nTraining classifier...")
    if HAS_XGBOOST:
        clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=args.seed,
            use_label_encoder=False,
            eval_metric="logloss",
        )
    else:
        print("  (XGBoost not available, using RandomForest)")
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=args.seed,
        )

    clf.fit(X_train_scaled, y_train)

    # Predict
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 60)
    mode_str = " [RANDOM WEIGHTS BASELINE]" if args.random_weights else ""
    print(f"RESULTS (Segment-level){mode_str}")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"AUC:      {auc:.3f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["HC", "Impaired"]))

    # Subject-level aggregation
    print("\n" + "=" * 60)
    print("RESULTS (Subject-level)")
    print("=" * 60)

    test_subject_ids = [subject_ids[i] for i in range(len(subject_ids)) if test_mask[i]]

    subj_probs = {}
    subj_labels = {}
    for i, s in enumerate(test_subject_ids):
        if s not in subj_probs:
            subj_probs[s] = []
            subj_labels[s] = y_test[i]
        subj_probs[s].append(y_prob[i])

    # Average probabilities per subject
    subj_mean_prob = np.array([np.mean(subj_probs[s]) for s in subj_probs])
    subj_true = np.array([subj_labels[s] for s in subj_probs])
    subj_pred = (subj_mean_prob > 0.5).astype(int)

    subj_accuracy = accuracy_score(subj_true, subj_pred)
    try:
        subj_auc = roc_auc_score(subj_true, subj_mean_prob)
    except ValueError:
        subj_auc = np.nan
        print("  (Cannot compute AUC - single class in test set)")

    print(f"Subjects in test: {len(subj_probs)}")
    print(f"Accuracy: {subj_accuracy:.3f}")
    print(f"AUC:      {subj_auc:.3f}")

    for s in sorted(subj_probs.keys()):
        true_label = "HC" if subj_labels[s] == 0 else "Impaired"
        pred_label = "HC" if np.mean(subj_probs[s]) < 0.5 else "Impaired"
        correct = "✓" if true_label == pred_label else "✗"
        print(f"  {s}: true={true_label:8s}, pred={pred_label:8s}, prob={np.mean(subj_probs[s]):.3f} {correct}")

    # Generate plots
    print("\nGenerating plots...")

    # ROC curve
    model_type = "RANDOM WEIGHTS" if args.random_weights else "Trained"
    title_suffix = f"\nRR={args.rr_target*100:.0f}%, Theiler={theiler}, Model={model_type}"
    plot_roc_curve(y_test, y_prob, output_dir, title_suffix, not args.no_show)

    # Feature importance
    if HAS_XGBOOST:
        importances = clf.feature_importances_
    else:
        importances = clf.feature_importances_
    plot_feature_importance(feature_names, importances, output_dir, not args.no_show)

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, output_dir, not args.no_show)

    print(f"\nAll plots saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
