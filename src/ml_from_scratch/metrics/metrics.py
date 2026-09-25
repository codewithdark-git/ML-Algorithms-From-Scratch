"""Evaluation metrics for regression and classification."""

import numpy as np
from typing import Optional


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R^2 (coefficient of determination).
    
    Best possible score is 1.0. Can be negative if model is worse than
    predicting the mean.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1 - ss_res / ss_tot


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy classification score."""
    return np.mean(y_true == y_pred)


def precision_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "binary",
    zero_division: float = 0.0,
) -> float:
    """
    Precision score.
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    average : str, default="binary"
        Averaging method: "binary", "micro", "macro", "weighted".
    zero_division : float, default=0.0
        Value to return when there is zero division.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if average == "binary":
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        if tp + fp == 0:
            return zero_division
        return tp / (tp + fp)
    elif average == "micro":
        return accuracy_score(y_true, y_pred)
    elif average == "macro":
        classes = np.unique(np.concatenate([y_true, y_pred]))
        precisions = []
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            if tp + fp == 0:
                precisions.append(zero_division)
            else:
                precisions.append(tp / (tp + fp))
        return np.mean(precisions)
    elif average == "weighted":
        classes = np.unique(np.concatenate([y_true, y_pred]))
        precisions = []
        supports = []
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            support = np.sum(y_true == c)
            if tp + fp == 0:
                precisions.append(zero_division)
            else:
                precisions.append(tp / (tp + fp))
            supports.append(support)
        return np.average(precisions, weights=supports)
    else:
        raise ValueError(f"Unknown average: {average}")


def recall_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "binary",
    zero_division: float = 0.0,
) -> float:
    """Recall score."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if average == "binary":
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        if tp + fn == 0:
            return zero_division
        return tp / (tp + fn)
    elif average == "micro":
        return accuracy_score(y_true, y_pred)
    elif average == "macro":
        classes = np.unique(np.concatenate([y_true, y_pred]))
        recalls = []
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            if tp + fn == 0:
                recalls.append(zero_division)
            else:
                recalls.append(tp / (tp + fn))
        return np.mean(recalls)
    elif average == "weighted":
        classes = np.unique(np.concatenate([y_true, y_pred]))
        recalls = []
        supports = []
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            support = np.sum(y_true == c)
            if tp + fn == 0:
                recalls.append(zero_division)
            else:
                recalls.append(tp / (tp + fn))
            supports.append(support)
        return np.average(recalls, weights=supports)
    else:
        raise ValueError(f"Unknown average: {average}")


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "binary",
    zero_division: float = 0.0,
) -> float:
    """F1 score."""
    prec = precision_score(y_true, y_pred, average=average, zero_division=zero_division)
    rec = recall_score(y_true, y_pred, average=average, zero_division=zero_division)
    
    if prec + rec == 0:
        return zero_division
    return 2 * prec * rec / (prec + rec)


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    n_labels = len(labels)
    cm = np.zeros((n_labels, n_labels), dtype=int)
    
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    for yt, yp in zip(y_true, y_pred):
        if yt in label_to_idx and yp in label_to_idx:
            cm[label_to_idx[yt], label_to_idx[yp]] += 1
    
    return cm


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    ROC AUC score for binary classification.
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True binary labels.
    y_score : array-like, shape (n_samples,)
        Target scores (probability estimates of positive class).
        
    Returns
    -------
    auc : float
        Area under the ROC curve.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    # Sort by score descending
    desc_score_indices = np.argsort(y_score)[::-1]
    y_true = y_true[desc_score_indices]
    y_score = y_score[desc_score_indices]
    
    # Count positives and negatives
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    # Compute ROC curve
    tpr = 0.0
    fpr = 0.0
    auc = 0.0
    prev_fpr = 0.0
    
    for i in range(len(y_true)):
        if y_true[i] == 1:
            tpr += 1.0 / n_pos
        else:
            fpr += 1.0 / n_neg
            auc += tpr * (fpr - prev_fpr)
            prev_fpr = fpr
    
    return auc


def silhouette_score(X: np.ndarray, labels: np.ndarray, metric: str = 'euclidean') -> float:
    """
    Silhouette score for clustering evaluation.
    
    The silhouette score measures how similar a sample is to its own cluster
    compared to other clusters. Range is [-1, 1] where 1 means well-separated
    clusters and -1 means misclassified samples.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.
    labels : array-like, shape (n_samples,)
        Cluster labels for each sample.
    metric : str, default='euclidean'
        Distance metric to use.
        
    Returns
    -------
    score : float
        Average silhouette score across all samples.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters == 1 or n_clusters == n_samples:
        return 0.0
    
    # Compute pairwise distances
    if metric == 'euclidean':
        # Efficient Euclidean distance computation
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x*y
        X_norm = np.sum(X ** 2, axis=1)
        distances = X_norm[:, np.newaxis] + X_norm[np.newaxis, :] - 2 * X @ X.T
        distances = np.maximum(distances, 0)  # Numerical stability
        distances = np.sqrt(distances)
    else:
        # Fallback to pairwise distance computation
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(X, metric=metric))
    
    silhouette_vals = np.zeros(n_samples)
    
    for i in range(n_samples):
        label_i = labels[i]
        
        # Average distance to points in same cluster (a_i)
        same_cluster = labels == label_i
        same_cluster[i] = False  # Exclude self
        if np.any(same_cluster):
            a_i = np.mean(distances[i, same_cluster])
        else:
            a_i = 0
        
        # Average distance to points in nearest other cluster (b_i)
        b_i = np.inf
        for other_label in unique_labels:
            if other_label == label_i:
                continue
            other_cluster = labels == other_label
            if np.any(other_cluster):
                mean_dist = np.mean(distances[i, other_cluster])
                if mean_dist < b_i:
                    b_i = mean_dist
        
        # Silhouette coefficient for sample i
        if max(a_i, b_i) > 0:
            silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i)
        else:
            silhouette_vals[i] = 0
    
    return np.mean(silhouette_vals)


def silhouette_samples(X: np.ndarray, labels: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """
    Silhouette coefficients for each sample.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.
    labels : array-like, shape (n_samples,)
        Cluster labels for each sample.
    metric : str, default='euclidean'
        Distance metric to use.
        
    Returns
    -------
    silhouette_vals : ndarray, shape (n_samples,)
        Silhouette coefficient for each sample.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters == 1 or n_clusters == n_samples:
        return np.zeros(n_samples)
    
    # Compute pairwise distances
    if metric == 'euclidean':
        X_norm = np.sum(X ** 2, axis=1)
        distances = X_norm[:, np.newaxis] + X_norm[np.newaxis, :] - 2 * X @ X.T
        distances = np.maximum(distances, 0)
        distances = np.sqrt(distances)
    else:
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(X, metric=metric))
    
    silhouette_vals = np.zeros(n_samples)
    
    for i in range(n_samples):
        label_i = labels[i]
        
        same_cluster = labels == label_i
        same_cluster[i] = False
        if np.any(same_cluster):
            a_i = np.mean(distances[i, same_cluster])
        else:
            a_i = 0
        
        b_i = np.inf
        for other_label in unique_labels:
            if other_label == label_i:
                continue
            other_cluster = labels == other_label
            if np.any(other_cluster):
                mean_dist = np.mean(distances[i, other_cluster])
                if mean_dist < b_i:
                    b_i = mean_dist
        
        if max(a_i, b_i) > 0:
            silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i)
        else:
            silhouette_vals[i] = 0
    
    return silhouette_vals


def adjusted_rand_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Adjusted Rand Index (ARI) for clustering evaluation.
    
    The ARI measures the similarity between two clusterings, adjusted for chance.
    Range is [-1, 1] where 1 means perfect match, 0 means random labeling,
    and negative values mean worse than random.
    
    Parameters
    ----------
    labels_true : array-like, shape (n_samples,)
        Ground truth cluster labels.
    labels_pred : array-like, shape (n_samples,)
        Predicted cluster labels.
        
    Returns
    -------
    ari : float
        Adjusted Rand Index.
    """
    labels_true = np.asarray(labels_true).ravel()
    labels_pred = np.asarray(labels_pred).ravel()
    
    n_samples = len(labels_true)
    
    # Contingency table
    classes_true = np.unique(labels_true)
    classes_pred = np.unique(labels_pred)
    
    n_true = len(classes_true)
    n_pred = len(classes_pred)
    
    # Create contingency matrix
    contingency = np.zeros((n_true, n_pred), dtype=int)
    
    for i in range(n_samples):
        t_idx = np.where(classes_true == labels_true[i])[0][0]
        p_idx = np.where(classes_pred == labels_pred[i])[0][0]
        contingency[t_idx, p_idx] += 1
    
    # Sum over rows and columns
    sum_rows = np.sum(contingency, axis=1)  # a_i
    sum_cols = np.sum(contingency, axis=0)  # b_j
    
    # Sum of combinations
    # sum(C(a_i, 2)) - sum of pairs in same true cluster
    sum_comb_rows = np.sum(sum_rows * (sum_rows - 1)) / 2
    # sum(C(b_j, 2)) - sum of pairs in same pred cluster
    sum_comb_cols = np.sum(sum_cols * (sum_cols - 1)) / 2
    # sum(C(n_ij, 2)) - sum of pairs in same true AND pred cluster
    sum_comb_contingency = np.sum(contingency * (contingency - 1)) / 2
    
    # Total pairs
    total_pairs = n_samples * (n_samples - 1) / 2
    
    # Expected index
    expected_index = sum_comb_rows * sum_comb_cols / total_pairs
    
    # Max index
    max_index = (sum_comb_rows + sum_comb_cols) / 2
    
    # ARI
    if max_index == expected_index:
        return 1.0
    
    ari = (sum_comb_contingency - expected_index) / (max_index - expected_index)
    
    return ari