import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity

class GraphBuilder:
    """
    A utility class to construct graph structures (Adjacency Matrices)
    from raw feature matrices using various similarity measures.
    """

    @staticmethod
    def build(Xs, method='knn', **kwargs):
        """
        Master function to build graphs for a list of feature matrices.
        
        Args:
            Xs (list of np.ndarray): List of feature matrices [X1, X2, ...].
            method (str): 'knn', 'rbf' (gaussian), or 'cosine'.
            **kwargs: Parameters specific to the method (e.g., k=10, sigma=1.0).
            
        Returns:
            list: List of adjacency matrices [A1, A2, ...].
        """
        graphs = []
        for i, X in enumerate(Xs):
            print(f"Building graph for View {i+1} using {method}...")
            
            if method == 'knn':
                adj = GraphBuilder.knn(X, **kwargs)
            elif method == 'rbf' or method == 'gaussian':
                adj = GraphBuilder.rbf(X, **kwargs)
            elif method == 'cosine':
                adj = GraphBuilder.cosine(X, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")
                
            graphs.append(adj)
        return graphs

    @staticmethod
    def knn(X, k=10, metric='euclidean', mode='connectivity', include_self=False):
        """
        Constructs a K-Nearest Neighbors graph.
        
        Args:
            k (int): Number of neighbors.
            metric (str): Distance metric ('euclidean', 'cosine', etc.).
            mode (str): 'connectivity' (0/1) or 'distance' (weighted).
            include_self (bool): If True, A[i,i] = 1.
        """
        A = kneighbors_graph(X, n_neighbors=k, metric=metric, mode=mode, include_self=include_self)
        return A

    @staticmethod
    def rbf(X, sigma=1.0, threshold=None, k=None):
        """
        Constructs a Gaussian (RBF) Kernel graph.
        A_ij = exp(- ||x_i - x_j||^2 / (2 * sigma^2))
        """
        gamma = 1.0 / (2.0 * sigma**2)
        A = rbf_kernel(X, gamma=gamma)
        
        # Sparsification
        if threshold is not None:
            A[A < threshold] = 0.0
            
        if k is not None:
            # Keep only top-k values per row to ensure sparsity
            # (Simple implementation: brute-force per row)
            for i in range(A.shape[0]):
                # Indices of values that are NOT in the top-k
                # argsort gives ascending, so we take the start of the array
                lower_indices = np.argsort(A[i])[:-(k+1)]
                A[i][lower_indices] = 0.0

        # Convert to sparse for efficiency
        return sp.csr_matrix(A)

    @staticmethod
    def cosine(X, k=None, threshold=None):
        """
        Constructs a Cosine Similarity graph.
        Great for text data (Cora, ACM, high-dim sparse vectors).
        """
        # 1. Compute Cosine Similarity (-1 to 1)
        A = cosine_similarity(X)
        
        # 2. Keep only positive correlations
        A[A < 0] = 0
        
        # 3. Sparsify
        if threshold is not None:
            A[A < threshold] = 0
            
        if k is not None:
            for i in range(A.shape[0]):
                lower_indices = np.argsort(A[i])[:-(k+1)]
                A[i][lower_indices] = 0.0
                
        return sp.csr_matrix(A)