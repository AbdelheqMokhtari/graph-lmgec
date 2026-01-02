"""
Clustering metrics used in the original LMGEC implementation.

This mirrors the behavior of:
- clustering_accuracy
- clustering_f1_score

from the original utils.py, plus a helper evaluate_clustering().
"""

from __future__ import annotations
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
import numpy as np

def ordered_confusion_matrix(y_true, y_pred):
    conf_mat = metrics.confusion_matrix(y_true, y_pred)
    w = np.max(conf_mat) - conf_mat
    row_ind, col_ind = linear_sum_assignment(w)
    conf_mat = conf_mat[row_ind, :]
    conf_mat = conf_mat[:, col_ind]
    return conf_mat

def cmat_to_psuedo_y_true_and_y_pred(cmat):
    y_true = []
    y_pred = []
    for true_class, row in enumerate(cmat):
        for pred_class, elm in enumerate(row):
            y_true.extend([true_class] * elm)
            y_pred.extend([pred_class] * elm)
    return y_true, y_pred
    
def clustering_accuracy(y_true, y_pred):
    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    return np.trace(conf_mat) / np.sum(conf_mat)

def clustering_f1_score(y_true, y_pred, **kwargs):
    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    pseudo_y_true, pseudo_y_pred = cmat_to_psuedo_y_true_and_y_pred(conf_mat)
    return metrics.f1_score(pseudo_y_true, pseudo_y_pred, **kwargs)


def evaluate_clustering(y_true, y_pred, average: str = "macro"):
    """Compute a small set of clustering metrics and return them as a dict.

    This is a convenience wrapper used by higher-level scripts/tests.
    """
    acc = clustering_accuracy(y_true, y_pred)
    f1 = clustering_f1_score(y_true, y_pred, average=average)
    return {"acc": acc, "f1": f1}


def evaluate_clustering(y_pred, y_true=None, X=None, average="macro"):
    """
    Compute clustering metrics (External & Internal).
    """
    scores = {}

    # A. External Validation (Needs Ground Truth)
    if y_true is not None:
        scores["ACC"] = clustering_accuracy(y_true, y_pred)
        scores["F1"] = clustering_f1_score(y_true, y_pred, average=average)
        scores["NMI"] = metrics.normalized_mutual_info_score(y_true, y_pred)
        scores["ARI"] = metrics.adjusted_rand_score(y_true, y_pred)

    # B. Internal Validation (Needs Embeddings X)
    if X is not None:
        # Avoid expensive silhouette on huge datasets
        if X.shape[0] < 20000:
            scores["Silhouette"] = metrics.silhouette_score(X, y_pred)
        
        scores["CH_Score"] = metrics.calinski_harabasz_score(X, y_pred)
        scores["DB_Score"] = metrics.davies_bouldin_score(X, y_pred)

    return scores
