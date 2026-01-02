import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity

def build_knn_graph(X, k=10, metric='euclidean', mode='connectivity', include_self=False):
    """
    Constructs a k-Nearest Neighbors graph.
    
    Args:
        X (array-like): Feature matrix of shape (n_samples, n_features).
        k (int): Number of neighbors to consider for each node.
        metric (str): Distance metric ('euclidean', 'cosine', 'minkowski', etc.).
        mode (str): 'connectivity' (0/1) or 'distance' (actual distances).
        include_self (bool): If True, A[i,i] = 1.
        
    Returns:
        sp.csr_matrix: Sparse adjacency matrix.
    """
    # sklearn includes the point itself as a neighbor in some versions, 
    # so we ask for k, then ensure diagonal is handled manually.
    A = kneighbors_graph(X, n_neighbors=k, mode=mode, metric=metric, include_self=include_self)
    
    # Ensure it is a symmetric graph (optional but recommended for spectral clustering)
    # A = 0.5 * (A + A.T) 
    
    return A

def build_gaussian_graph(X, sigma=1.0, threshold=0.0, k=None):
    """
    Constructs a graph using Gaussian (RBF) Kernel similarity.
    Formula: A_ij = exp( - ||x_i - x_j||^2 / (2 * sigma^2) )
    
    Args:
        X (array-like): Feature matrix.
        sigma (float): Bandwidth parameter. Controls how quickly similarity drops.
        threshold (float): Edges with weight < threshold are set to 0 (sparsification).
        k (int): If not None, keeps only the top-k weights per node (sparsification).
        
    Returns:
        sp.csr_matrix: Sparse adjacency matrix.
    """
    gamma = 1.0 / (2.0 * sigma**2)
    A_dense = rbf_kernel(X, gamma=gamma)
    
    # Remove self-loops (diagonal = 1.0) usually needed for GNNs that add it later
    np.fill_diagonal(A_dense, 0.0)
    
    if k is not None:
        # Keep only top-k neighbors
        nb_samples = A_dense.shape[0]
        for i in range(nb_samples):
            # argsort gives indices of smallest to largest. We want largest.
            # We take the last k elements
            top_k_idx = np.argsort(A_dense[i])[-k:]
            
            # Create a mask to zero out everything else
            mask = np.zeros(nb_samples, dtype=bool)
            mask[top_k_idx] = True
            A_dense[i][~mask] = 0.0

    if threshold > 0:
        A_dense[A_dense < threshold] = 0.0
        
    return sp.csr_matrix(A_dense)

def build_cosine_graph(X, k=None, threshold=0.0):
    """
    Constructs a graph based on Cosine Similarity.
    Good for text data (Bag of Words, TF-IDF).
    
    Args:
        X (array-like): Feature matrix.
        k (int): Keep top-k similar neighbors.
        threshold (float): Zero out similarities below this value.
        
    Returns:
        sp.csr_matrix: Sparse adjacency matrix.
    """
    A_dense = cosine_similarity(X)
    np.fill_diagonal(A_dense, 0.0)
    
    if k is not None:
        nb_samples = A_dense.shape[0]
        for i in range(nb_samples):
            top_k_idx = np.argsort(A_dense[i])[-k:]
            mask = np.zeros(nb_samples, dtype=bool)
            mask[top_k_idx] = True
            A_dense[i][~mask] = 0.0
            
    if threshold > 0:
        A_dense[A_dense < threshold] = 0.0
        
    return sp.csr_matrix(A_dense)

def generate_multiview_graphs(Xs, method='knn', **kwargs):
    """
    Helper to process a list of feature matrices into a list of graphs.
    
    Args:
        Xs (list of arrays): List of feature matrices [X1, X2, ...]
        method (str): 'knn', 'gaussian', or 'cosine'
        **kwargs: Arguments passed to the builder (e.g., k=10, sigma=1.0)
        
    Returns:
        list of sp.csr_matrix: The generated adjacency matrices.
    """
    builders = {
        'knn': build_knn_graph,
        'gaussian': build_gaussian_graph,
        'cosine': build_cosine_graph
    }
    
    if method not in builders:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(builders.keys())}")
        
    builder_func = builders[method]
    return [builder_func(X, **kwargs) for X in Xs]