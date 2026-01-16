"""Classification pipeline for MCI vs HC discrimination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    cohen_kappa_score,
)
from omegaconf import DictConfig

import xgboost as xgb


@dataclass
class ClassificationResults:
    """Container for classification results."""

    accuracy: float
    auc: float
    kappa: float
    report: str

    # Per-fold results
    fold_accuracies: np.ndarray
    fold_aucs: np.ndarray

    # Predictions
    predictions: np.ndarray
    probabilities: np.ndarray

    def __str__(self) -> str:
        return (
            f"Classification Results:\n"
            f"  Accuracy: {self.accuracy:.4f} (std: {self.fold_accuracies.std():.4f})\n"
            f"  AUC: {self.auc:.4f} (std: {self.fold_aucs.std():.4f})\n"
            f"  Kappa: {self.kappa:.4f}\n"
            f"\n{self.report}"
        )


class ClassificationPipeline:
    """
    Classification pipeline with proper cross-validation.

    CRITICAL: Uses GroupKFold to prevent label leakage at subject level.

    Args:
        cfg: Configuration with classification settings
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.n_splits = cfg.classification.cv_folds

        # Select classifier
        model_name = cfg.classification.model
        if model_name == "xgboost":
            self.classifier = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=cfg.experiment.seed,
                use_label_encoder=False,
                eval_metric="logloss",
            )
        else:
            raise ValueError(f"Unknown classifier: {model_name}")

        # Build pipeline with scaling
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", self.classifier),
        ])

    def evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        subject_ids: np.ndarray,
        evaluation_level: Literal["segment", "subject"] = "subject",
    ) -> ClassificationResults:
        """
        Evaluate classifier with proper cross-validation.

        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Class labels (n_samples,)
            subject_ids: Subject identifiers for GroupKFold
            evaluation_level: "segment" or "subject" level metrics

        Returns:
            ClassificationResults
        """
        # Use GroupKFold to prevent subject leakage
        cv = GroupKFold(n_splits=self.n_splits)

        fold_accuracies = []
        fold_aucs = []
        all_predictions = np.zeros_like(labels)
        all_probabilities = np.zeros(len(labels))

        for fold, (train_idx, test_idx) in enumerate(cv.split(features, labels, groups=subject_ids)):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            # Train
            self.pipeline.fit(X_train, y_train)

            # Predict
            predictions = self.pipeline.predict(X_test)
            probabilities = self.pipeline.predict_proba(X_test)[:, 1]

            all_predictions[test_idx] = predictions
            all_probabilities[test_idx] = probabilities

            if evaluation_level == "subject":
                # Aggregate to subject level
                test_subjects = subject_ids[test_idx]
                subj_pred, subj_true, subj_probs = self._aggregate_to_subject(
                    predictions, y_test, probabilities, test_subjects
                )
                fold_acc = accuracy_score(subj_true, subj_pred)
                # CRITICAL FIX: Use probabilities (not hard labels) for AUC
                try:
                    fold_auc = roc_auc_score(subj_true, subj_probs)
                except ValueError:
                    # Single-class fold
                    fold_auc = np.nan
            else:
                fold_acc = accuracy_score(y_test, predictions)
                fold_auc = roc_auc_score(y_test, probabilities)

            fold_accuracies.append(fold_acc)
            fold_aucs.append(fold_auc)

        # Overall metrics
        if evaluation_level == "subject":
            subj_pred, subj_true, subj_probs = self._aggregate_to_subject(
                all_predictions, labels, all_probabilities, subject_ids
            )
            accuracy = accuracy_score(subj_true, subj_pred)
            # CRITICAL FIX: Use probabilities (not hard labels) for AUC
            auc = roc_auc_score(subj_true, subj_probs)
            kappa = cohen_kappa_score(subj_true, subj_pred)
            report = classification_report(subj_true, subj_pred, target_names=["HC", "MCI"])
        else:
            accuracy = accuracy_score(labels, all_predictions)
            auc = roc_auc_score(labels, all_probabilities)
            kappa = cohen_kappa_score(labels, all_predictions)
            report = classification_report(labels, all_predictions, target_names=["HC", "MCI"])

        return ClassificationResults(
            accuracy=accuracy,
            auc=auc,
            kappa=kappa,
            report=report,
            fold_accuracies=np.array(fold_accuracies),
            fold_aucs=np.array(fold_aucs),
            predictions=all_predictions,
            probabilities=all_probabilities,
        )

    def _aggregate_to_subject(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: np.ndarray,
        subject_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aggregate segment-level predictions to subject level.

        Returns:
            subject_predictions: Hard predictions (majority vote)
            subject_labels: Ground truth labels
            subject_probs: Mean probability per subject (for AUC)
        """
        unique_subjects = np.unique(subject_ids)
        subject_predictions = []
        subject_labels = []
        subject_probs = []

        for subj in unique_subjects:
            mask = subject_ids == subj

            # Majority vote for hard prediction
            pred = int(predictions[mask].mean() >= 0.5)
            subject_predictions.append(pred)

            # Mean probability for AUC (CRITICAL FIX)
            prob = probabilities[mask].mean()
            subject_probs.append(prob)

            # Ground truth (should be same for all segments)
            subject_labels.append(labels[mask][0])

        return np.array(subject_predictions), np.array(subject_labels), np.array(subject_probs)

    def compare_baseline_vs_state_conditioned(
        self,
        baseline_features: np.ndarray,
        state_features: np.ndarray,
        labels: np.ndarray,
        subject_ids: np.ndarray,
    ) -> dict:
        """
        Compare baseline (whole-recording) vs state-conditioned features.

        This is the key integration experiment.

        Returns:
            Dictionary with comparison metrics
        """
        # Evaluate baseline
        baseline_results = self.evaluate(
            baseline_features, labels, subject_ids, evaluation_level="subject"
        )

        # Evaluate state-conditioned
        state_results = self.evaluate(
            state_features, labels, subject_ids, evaluation_level="subject"
        )

        # Compute improvements
        auc_improvement = state_results.auc - baseline_results.auc
        variance_ratio = state_results.fold_aucs.std() / (baseline_results.fold_aucs.std() + 1e-8)

        return {
            "baseline": baseline_results,
            "state_conditioned": state_results,
            "auc_improvement": auc_improvement,
            "variance_ratio": variance_ratio,
            "success": (
                auc_improvement >= self.cfg.success_criteria.min_auc_improvement
                and variance_ratio <= self.cfg.success_criteria.max_cv_variance_ratio
            ),
        }
