"""
Clustering evaluation metrics for LMGEC (and other clustering algorithms).

This module provides:
- clustering_accuracy: ACC with best label permutation (Hungarian)
- clustering_f1_score: F1 score with optional label alignment
- evaluate_clustering: convenience wrapper returning all metrics in a dict
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
)


def _best_label_permutation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Find the best permutation of predicted labels to match true labels,
    using the Hungarian algorithm on the confusion matrix.

    Returns a new y_pred_aligned array.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Build confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 0:
        # No labels; return original
        return y_pred

    # Hungarian algorithm: maximize trace of cm
    # => minimize cost = max(cm) - cm
    cost_matrix = cm.max() - cm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build mapping from predicted label -> aligned label
    mapping = {pred_label: true_label for true_label, pred_label in zip(row_ind, col_ind)}

    # Apply mapping to y_pred
    y_pred_aligned = np.vectorize(lambda c: mapping.get(c, c))(y_pred)
    return y_pred_aligned


def clustering_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    align_labels: bool = True,
) -> float:
    """
    Clustering accuracy (ACC), optionally after best label permutation.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth labels.
    y_pred : array-like, shape (n_samples,)
        Predicted cluster labels.
    align_labels : bool, default=True
        If True, apply best label permutation before computing accuracy.

    Returns
    -------
    acc : float
        Clustering accuracy in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if align_labels:
        y_pred = _best_label_permutation(y_true, y_pred)

    return (y_true == y_pred).mean()


def clustering_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
    align_labels: bool = True,
) -> float:
    """
    Clustering F1 score, optionally after best label permutation.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth labels.
    y_pred : array-like, shape (n_samples,)
        Predicted cluster labels.
    average : str, default="macro"
        F1 averaging method (passed to sklearn.metrics.f1_score).
    align_labels : bool, default=True
        If True, apply best label permutation before computing F1.

    Returns
    -------
    f1 : float
        F1 score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if align_labels:
        y_pred = _best_label_permutation(y_true, y_pred)

    return f1_score(y_true, y_pred, average=average)


def evaluate_clustering(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loss: Optional[float] = None,
    average: str = "macro",
    align_labels: bool = True,
) -> Dict[str, float]:
    """
    Convenience function to compute all metrics the original code used.

    Includes:
    - acc: clustering accuracy
    - f1: F1 score (macro by default)
    - nmi: normalized mutual information
    - ari: adjusted Rand index
    - loss: optionally pass the final loss from LMGEC (if available)

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth labels.
    y_pred : array-like, shape (n_samples,)
        Predicted cluster labels.
    loss : float, optional
        Final objective value (if you want to track it alongside metrics).
    average : str, default="macro"
        F1 averaging method.
    align_labels : bool, default=True
        If True, align labels by best permutation for ACC/F1.
        For NMI/ARI this is not needed and they are computed on the raw labels.

    Returns
    -------
    metrics : dict
        Dictionary with keys 'acc', 'f1', 'nmi', 'ari', and optionally 'loss'.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # ACC and F1 with optional alignment
    acc = clustering_accuracy(y_true, y_pred, align_labels=align_labels)
    f1 = clustering_f1_score(y_true, y_pred, average=average, align_labels=align_labels)

    # NMI and ARI (label permutation doesn't matter for these)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)

    metrics = {
        "acc": float(acc),
        "f1": float(f1),
        "nmi": float(nmi),
        "ari": float(ari),
    }
    if loss is not None:
        metrics["loss"] = float(loss)

    return metrics
