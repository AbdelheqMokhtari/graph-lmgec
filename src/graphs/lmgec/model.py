"""
LMGEC model definition.

implementation of the LMGEC (Linear Multi-view Graph Embedding Clustering) algorithm.
This module exposes:

- LMGEC: estimator with fit / fit_predict and accessors.
"""

import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import numpy as np
from .utils import get_propagated_features
from .metrics import evaluate_clustering
import warnings

# Add 'metrics' from sklearn
from sklearn import metrics 

# Update your local import to include the specific matching functions
from .metrics import evaluate_clustering, clustering_accuracy, clustering_f1_score

def _update_rule_F(XW, G, K):

    F = tf.math.unsorted_segment_mean(XW, G, num_segments=K)
    return F

def _update_rule_W(X, F, G):

    _, U, V = tf.linalg.svd(tf.transpose(X) @ tf.gather(F, G), full_matrices=False)
    W = U @ tf.transpose(V)
    return W

def _update_rule_G(XW, F):
    centroids_expanded = F[:,None,...]
    distances = tf.reduce_mean(tf.math.squared_difference(XW, centroids_expanded), 2)
    G = tf.math.argmin(distances, 0, output_type=tf.dtypes.int32)
    return G

@tf.function
def _train_loop(Xs, F, G, alphas, k, max_iter):
    n_views = len(Xs)
    losses = tf.TensorArray(tf.float64, size=0, dynamic_size=True)
    prev_loss = tf.float64.max

    for i in tf.range(max_iter):
        loss = 0
        XW_consensus = 0
        for v in range(n_views):
            Wv = _update_rule_W(Xs[v], F, G)
            XWv = Xs[v]@Wv
            XW_consensus += alphas[v] * XWv
            loss_v = tf.linalg.norm(Xs[v] - tf.gather(F @ tf.transpose(Wv), G))
            loss += alphas[v] * loss_v
        G = _update_rule_G(XW_consensus, F)
        F = _update_rule_F(XW_consensus, G, k)

        losses = losses.write(i, loss)
        prev_loss = loss ## i'll check this later

    return G, F, XW_consensus, losses.stack()


class LMGEC:
    def __init__(
        self,
        n_clusters: int,
        embedding_dim: int,
        temperature: float = 10.0,
        max_iter: int = 30,
        tol: float = 1e-4,
        beta: float = 1.0,
        center_data=True,
        tf_idf=False,
        scale=False
    ) -> None:
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.max_iter = max_iter
        self.tol = tol
        self.beta = beta
        self.center_data = center_data
        self.tf_idf = tf_idf
        self.scale = scale
        self.G_ = None
        self.F_ = None
        self.XW_consensus_ = None
        self.y_true_ = None
        self.alphas_ = None
        self.loss_history_ = None

    def _init_W(self, X):
        """Internal helper to initialize W using SVD."""
        n_features = X.shape[1]
        
        # Check if requested embedding is too large
        if self.embedding_dim >= n_features:
            warnings.warn(
                f"View has {n_features} features, but embedding_dim is {self.embedding_dim}. "
                "Skipping SVD reduction and keeping original features for this view.",
                UserWarning
            )
            # Return Identity matrix (keeps features as they are)
            return np.eye(n_features)

        svd = TruncatedSVD(n_components=self.embedding_dim).fit(X)
        return svd.components_.T

    def _init_G_F(self, XW):
        """Internal helper to initialize G and F using KMeans."""
        # n_init=10 ensures stable K-Means initialization
        km = KMeans(n_clusters=self.n_clusters, n_init=10).fit(XW)
        G = km.labels_.astype(np.int32)
        F = km.cluster_centers_
        return G, F

    def fit(self, Xs, adjs=None, y_true =None):
        """
        Train the model.
        
        Args:
            Xs: List of feature matrices. 
                - If adjs is provided: specific raw features (X).
                - If adjs is None: specifies propagated features (H).
            adjs: (Optional) List of adjacency matrices. 
                  Pass None in the list for views without a graph structure.
        """

        if y_true is not None:
            self.y_true_ = np.array(y_true)

        if adjs is not None:
            if len(Xs) != len(adjs):
                raise ValueError(f"Mismatch: {len(Xs)} features vs {len(adjs)} graphs.")
            
            print("Adjacency matrices provided. Computing propagation...")
            new_Xs = []
            for X, A in zip(Xs, adjs):
                # Eq 6: H <- SX
                # This combines Graph + Features into the final matrix we use
                h_matrix = get_propagated_features(A, X, beta=self.beta, tf_idf=self.tf_idf, center=self.center_data, scale=self.scale)
                new_Xs.append(h_matrix)
            
            Xs = new_Xs
        
        # Ensure float64
        Xs = [x.astype(np.float64) for x in Xs]
        n_views = len(Xs)

        # inital G and F 
        alphas = np.zeros(n_views)

        XW_consensus = 0
        for v in range(n_views):
            Wv = self._init_W(Xs[v])
            XWv = Xs[v]@Wv
            Gv, Fv = self._init_G_F(XWv)
            inertia = np.linalg.norm(XWv - Fv[Gv])
            alphas[v] = np.exp(-inertia/self.temperature)
            XW_consensus += alphas[v] * XWv
        WX_consensus = XW_consensus / alphas.sum()
        G, F = self._init_G_F(XW_consensus)

        G, F, XW_consensus, loss_history = _train_loop(Xs, F, G, alphas, self.n_clusters, self.max_iter)
        
        self.G_ = G.numpy()
        self.F_ = F.numpy()
        self.XW_consensus_ = XW_consensus.numpy()
        self.alphas_ = alphas
        self.loss_history_ = loss_history.numpy()
        return self
    
    def evaluate(self, y_true=None):
        """
        Evaluate clustering quality.
        Uses metrics.py logic. Handles cases with and without ground truth.
        """
        if self.G_ is None:
            raise RuntimeError("Model not fitted.")
            
        target_labels = y_true if y_true is not None else self.y_true_
        
        results = evaluate_clustering(
            y_pred=self.G_,
            y_true=target_labels,     # Can be None
            X=self.XW_consensus_      # Used for Silhouette/CH/DB
        )
        
        print("\n--- Evaluation Results ---")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")
            
        return results

    # --- Helper for External Metrics ---
    def _get_labels(self, y_true):
        """Internal helper to resolve which labels to use."""
        if self.G_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
            
        target_labels = y_true if y_true is not None else self.y_true_
        
        if target_labels is None:
            raise ValueError(
                "Ground truth labels required for this metric. "
                "Pass 'y_true' or fit with labels."
            )
        return target_labels

    # --- External Metrics (Require Ground Truth) ---
    
    def accuracy(self, y_true=None):
        """Returns Clustering Accuracy (Hungarian matching)."""
        labels = self._get_labels(y_true)
        return clustering_accuracy(labels, self.G_)

    def f1_score(self, y_true=None, average="macro"):
        """Returns F1 Score (Hungarian matching)."""
        labels = self._get_labels(y_true)
        return clustering_f1_score(labels, self.G_, average=average)

    def nmi(self, y_true=None):
        """Returns Normalized Mutual Information."""
        labels = self._get_labels(y_true)
        return metrics.normalized_mutual_info_score(labels, self.G_)

    def ari(self, y_true=None):
        """Returns Adjusted Rand Index."""
        labels = self._get_labels(y_true)
        return metrics.adjusted_rand_score(labels, self.G_)

    # --- Internal Metrics (No Labels Needed) ---

    def silhouette(self):
        """Returns Silhouette Coefficient (Geometric quality)."""
        if self.XW_consensus_ is None:
            raise RuntimeError("Model not fitted.")
        if self.XW_consensus_.shape[0] > 20000:
            warnings.warn("Dataset is large; Silhouette score is expensive.", UserWarning)
        return metrics.silhouette_score(self.XW_consensus_, self.G_)

    def davies_bouldin(self):
        """Returns Davies-Bouldin Index (Lower is better)."""
        if self.XW_consensus_ is None:
            raise RuntimeError("Model not fitted.")
        return metrics.davies_bouldin_score(self.XW_consensus_, self.G_)
    
    def calinski_harabasz(self):
        """Returns Calinski-Harabasz Index (Higher is better)."""
        if self.XW_consensus_ is None:
            raise RuntimeError("Model not fitted.")
        return metrics.calinski_harabasz_score(self.XW_consensus_, self.G_)







