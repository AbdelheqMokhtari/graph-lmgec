from __future__ import annotations
from typing import Union

import scipy.sparse as sp
import numpy as np
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer



def preprocess_graph(adj, beta=1.0):
    """
    Computes the linear propagation matrix S (Eq 5).
    S = D^{-1}(A + beta*I)
    """
    # Ensure sparse for efficient calculation
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
        
    # Eq 4: Add self-loops
    adj = adj + beta * sp.eye(adj.shape[0])
    
    # Eq 5: Row Normalization (D^-1 * A)
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    S = r_mat_inv.dot(adj)
    
    return S




def normalize_features(
    features: Union[np.ndarray, sp.spmatrix],
    tf_idf: bool = False
) -> Union[np.ndarray, sp.spmatrix]:
    """
    Normalize feature vectors using L2 normalization, with optional TF-IDF weighting.

    If `tf_idf` is True, the input features are first transformed using TF-IDF
    weighting and then L2-normalized. Otherwise, the features are directly
    L2-normalized.

    This function supports both dense (NumPy arrays) and sparse
    (SciPy sparse matrices) feature representations.

    Parameters
    ----------
    features : np.ndarray or scipy.sparse.spmatrix
        Feature matrix of shape (n_samples, n_features). Can be dense or sparse.

    tf_idf : bool, optional (default=False)
        Whether to apply TF-IDF transformation before L2 normalization.
        Typically used for sparse count-based features.

    Returns
    -------
    np.ndarray or scipy.sparse.spmatrix
        L2-normalized feature matrix with the same shape and type
        (dense or sparse) as the input.

    Notes
    -----
    - When `tf_idf=True`, this function is intended for non-negative,
      count-based features.
    - L2 normalization scales each row to unit norm.
    """
    if tf_idf:
        features = TfidfTransformer(norm="l2").fit_transform(features)
    else:
        features = normalize(features, norm="l2")

    return features


def preprocess_features(features, center=False, scale=False):
    """
    Preprocess multiple feature matrices (views) by optionally centering
    and/or scaling each view independently.

    Parameters
    ----------
    features : list of numpy.ndarray or scipy.sparse.spmatrix
        List of feature matrices, one per view. Each matrix has shape
        (n_samples, n_features_v).

    center : bool, optional (default=False)
        Whether to center features by removing the mean (per view).

    scale : bool, optional (default=False)
        Whether to scale features to unit variance (per view).

    Returns
    -------
    list of numpy.ndarray or scipy.sparse.spmatrix
        List of transformed feature matrices. Each output matches the
        input type (dense or sparse) of its corresponding view.
    """

    if not (center or scale):
        return features

    processed = []

    for v, X in enumerate(features):

        # Sparse matrices cannot be mean-centered safely
        local_center = center
        if sp.issparse(X) and center:
            print(
                f"Warning (view {v}): Disabling centering on sparse data "
                "to preserve sparsity."
            )
            local_center = False

        scaler = StandardScaler(with_mean=local_center, with_std=scale)
        X_proc = scaler.fit_transform(X)

        processed.append(X_proc)

    return processed


def get_propagated_features(
    features,
    adj=None,
    beta=1.0,
    tf_idf=False,
):
    """
    Complete LMGEC Preprocessing Pipeline.
    
    Args:
        adj: Adjacency matrix. Pass None for Identity.
        features: Feature matrix.
        beta: Graph self-loop weight.
        tf_idf: Whether to use TF-IDF normalization on raw features.
    """
    x = normalize_features(features, tf_idf)

    if adj is None:
        return x

    S = preprocess_graph(adj, beta=beta)
    
    H = S.dot(x)

    # Converting the matrix into a dense matrix may crash the RAM 
    # the original idea behind it was when we do propagation the H it becomes sparse  

    if sp.issparse(H):
        H = H.toarray()
    elif isinstance(H, np.matrix):
        H = np.asarray(H)

    return H