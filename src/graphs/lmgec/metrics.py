"""
Clustering metrics used in the original LMGEC implementation.

This mirrors the behavior of:
- clustering_accuracy
- clustering_f1_score

from the original utils.py, plus a helper evaluate_clustering().
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics as sk_metrics
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
)


def ordered_confusion_matrix(y_true, y_pred):
    """
    Reorder confusion matrix using the Hungarian algorithm to best match labels.
    """
    conf_mat = sk_metrics.confusion_matrix(y_true, y_pred)
    if conf_mat.size == 0:
        return conf_mat

    w = np.max(conf_mat) - conf_mat
    row_ind, col_ind = linear_sum_assignment(w)
    conf_mat = conf_mat[row_ind, :]
    conf_mat = conf_mat[:, col_ind]
    return conf_mat


def cmat_to_pseudo_y_true_and_y_pred(cmat):
    """
    Expand a confusion matrix into pseudo y_true / y_pred label vectors.
    """
    y_true = []
    y_pred = []
    for true_class, row in enumerate(cmat):
        for pred_class, elm in enumerate(row):
            if elm > 0:
                y_true.extend([true_class] * elm)
                y_pred.extend([pred_class] * elm)
    return np.array(y_true), np.array(y_pred)


def clustering_accuracy(y_true, y_pred) -> float:
    """
    Clustering accuracy (ACC) with optimal label permutation.
    """
    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    if conf_mat.size == 0:
        return 0.0
    return float(np.trace(conf_mat) / np.sum(conf_mat))


def clustering_f1_score(y_true, y_pred, **kwargs) -> float:
    """
    Clustering F1 score, using the confusion-matrix expansion trick
    from the original implementation.
    """
    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    if conf_mat.size == 0:
        return 0.0
    pseudo_y_true, pseudo_y_pred = cmat_to_pseudo_y_true_and_y_pred(conf_mat)
    return float(sk_metrics.f1_score(pseudo_y_true, pseudo_y_pred, **kwargs))


def evaluate_clustering(
    y_true,
    y_pred,
    loss: float | None = None,
    average: str = "macro",
) -> dict:
    """
    Convenience wrapper to compute:
      - acc
      - f1
      - nmi
      - ari
      - (optional) loss
    """
    acc = clustering_accuracy(y_true, y_pred)
    f1 = clustering_f1_score(y_true, y_pred, average=average)
    nmi = float(normalized_mutual_info_score(y_true, y_pred))
    ari = float(adjusted_rand_score(y_true, y_pred))

    metrics = {
        "acc": acc,
        "f1": f1,
        "nmi": nmi,
        "ari": ari,
    }
    if loss is not None:
        metrics["loss"] = float(loss)
    return metrics

