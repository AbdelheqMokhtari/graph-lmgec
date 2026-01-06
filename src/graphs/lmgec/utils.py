from __future__ import annotations


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

def preprocess_features(features, tf_idf=False):
    """
    Normalizes features (TF-IDF or L2).
    """
    if tf_idf:
        features = TfidfTransformer(norm="l2").fit_transform(features)
    else: 
        features = normalize(features, norm="l2")
    return features

def get_propagated_features(adj, features, beta=1.0, tf_idf=False, center=True, scale=False):
    """
    Complete LMGEC Preprocessing Pipeline.
    
    Args:
        adj: Adjacency matrix. Pass None for Identity.
        features: Feature matrix.
        beta: Graph self-loop weight.
        tf_idf: Whether to use TF-IDF normalization on raw features.
        center: Whether to center data (Mean=0). LMGEC usually requires this.
        scale: Whether to scale data (Variance=1). "Centré-Réduit" if True.
    """

    # Centering / Scaling
    if center or scale:
        # with_mean=True -> Centré (Centered)
        # with_std=True  -> Réduit (Reduced/Scaled)
        scaler = StandardScaler(with_mean=center, with_std=scale)
        features = scaler.fit_transform(features)

    # Feature Normalization (L2 or TF-IDF)
    X = preprocess_features(features, tf_idf=tf_idf)

    # Propagation (H = SX)
    if adj is None:
        H = X
    else:
        S = preprocess_graph(adj, beta=beta)
        H = S.dot(X)

    # Densify
    if sp.issparse(H):
        H = H.toarray()
    elif isinstance(H, np.matrix):
        H = np.asarray(H)



    return H