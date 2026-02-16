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
    Preprocess a feature matrix by optionally centering and/or scaling.

    This function applies standardization to the input features using
    scikit-learn's ``StandardScaler``. Centering (mean removal) and scaling
    (unit variance) can be enabled independently. For sparse input matrices,
    centering is automatically disabled to preserve sparsity and avoid
    excessive memory usage.

    Parameters
    ----------
    features : numpy.ndarray or scipy.sparse.spmatrix
        Feature matrix of shape (n_samples, n_features). Can be dense or sparse.

    center : bool, optional (default=False)
        Whether to center features by removing the mean (mean = 0).
        This is commonly required for linear algebra–based methods
        (e.g., LMGEC).

    scale : bool, optional (default=False)
        Whether to scale features to unit variance (variance = 1).
        When both ``center`` and ``scale`` are True, the data is
        standardized ("centré-réduit").

    Returns
    -------
    numpy.ndarray or scipy.sparse.spmatrix
        The transformed feature matrix with the same shape as the input.
        The output type (dense or sparse) matches the input type.

    Notes
    -----
    - Mean-centering sparse matrices is not supported, as it would destroy
      sparsity and significantly increase memory usage. If ``center=True``
      and the input is sparse, centering is automatically disabled.
    - If both ``center`` and ``scale`` are False, the input features are
      returned unchanged.
    """
  
    if not (center or scale):
        return features

    # Sparse matrices cannot be mean-centered safely
    if sp.issparse(features) and center:
        # SAFE MODE: disable centering to preserve sparsity and memory
        print(
            "Warning: Disabling centering on sparse data to prevent memory issues."
        )
        center = False

    scaler = StandardScaler(with_mean=center, with_std=scale)
    features = scaler.fit_transform(features)

    return features

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

    # if sp.issparse(H):
    #    H = H.toarray()
    # elif isinstance(H, np.matrix):
    #    H = np.asarray(H)

    return H