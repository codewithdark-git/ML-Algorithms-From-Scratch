"""Evaluation metrics for regression, classification, and clustering."""

from ml_from_scratch.metrics.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    silhouette_score,
    silhouette_samples,
    adjusted_rand_score,
)

__all__ = [
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "roc_auc_score",
    "confusion_matrix",
    "silhouette_score",
    "silhouette_samples",
    "adjusted_rand_score",
]