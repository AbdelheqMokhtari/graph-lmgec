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
from .utils import get_propagated_features, preprocess_features
from .metrics import evaluate_clustering
import warnings
from sklearn import metrics 
from .metrics import evaluate_clustering, clustering_accuracy, clustering_f1_score
from scipy.io import savemat, loadmat
import os

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
    losses = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    prev_loss = tf.float64.max

    XW_consensus = tf.zeros(
        (tf.shape(Xs[0])[0], tf.shape(F)[1]),
        dtype=tf.float32
    )
    for i in tf.range(max_iter):
        loss = tf.constant(0.0, dtype=tf.float32)

        XW_consensus = tf.zeros_like(XW_consensus)
        
        for v in range(n_views):
            Wv = _update_rule_W(Xs[v], F, G)
            XWv = tf.linalg.matmul(Xs[v], Wv)
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
            self.y_true_ = np.asarray(y_true)

        # Validate adjacency list if provided
        if adjs is not None and len(Xs) != len(adjs):
            raise ValueError(
                f"Mismatch: {len(Xs)} feature matrices vs {len(adjs)} adjacency matrices."
            )

        new_Xs = []

        for i, X in enumerate(Xs):
            A = adjs[i] if adjs is not None else None

            # Eq 6: H <- S X (or identity if A is None)
            H = get_propagated_features(
                features=X,
                adj=A,
                beta=self.beta,
                tf_idf=self.tf_idf,
            )

            new_Xs.append(H)

        Xs = preprocess_features(new_Xs, center=self.center_data, scale=self.scale)

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
        

        # using Safe Fallback (Uniform weighting) / if all views collapse means they are all equally bad
        alpha_sum = alphas.sum()
        if alpha_sum == 0:
            alphas[:] = 1.0 / len(alphas)
            alpha_sum = 1.0

        XW_consensus /= alpha_sum

        G, F = self._init_G_F(XW_consensus)

        Xs_tf = [tf.convert_to_tensor(X, dtype=tf.float32) for X in Xs]
        F_tf = tf.convert_to_tensor(F, dtype=tf.float32)
        G_tf = tf.convert_to_tensor(G, dtype=tf.int32)
        alphas_tf = tf.convert_to_tensor(alphas, dtype=tf.float32)

        G, F, XW_consensus, loss_history = _train_loop(
            Xs_tf, F_tf, G_tf, alphas_tf,
            self.n_clusters, self.max_iter
        )


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
            y_true=target_labels,     
            X=self.XW_consensus_      
        )
        
        print("\n--- Evaluation Results ---")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")
            
        return results

    # Helper for External Metrics 
    def _get_labels(self, y_true):

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
        labels = self._get_labels(y_true)
        return clustering_accuracy(labels, self.G_)

    def f1_score(self, y_true=None, average="macro"):
        labels = self._get_labels(y_true)
        return clustering_f1_score(labels, self.G_, average=average)

    def nmi(self, y_true=None):
        labels = self._get_labels(y_true)
        return metrics.normalized_mutual_info_score(labels, self.G_)

    def ari(self, y_true=None):
        labels = self._get_labels(y_true)
        return metrics.adjusted_rand_score(labels, self.G_)

    # --- Internal Metrics (No Labels Needed) ---

    def silhouette(self):
        if self.XW_consensus_ is None:
            raise RuntimeError("Model not fitted.")
        if self.XW_consensus_.shape[0] > 20000:
            warnings.warn("Dataset is large; Silhouette score is expensive.", UserWarning)
        return metrics.silhouette_score(self.XW_consensus_, self.G_)

    def davies_bouldin(self):
        if self.XW_consensus_ is None:
            raise RuntimeError("Model not fitted.")
        return metrics.davies_bouldin_score(self.XW_consensus_, self.G_)
    
    def calinski_harabasz(self):
        if self.XW_consensus_ is None:
            raise RuntimeError("Model not fitted.")
        return metrics.calinski_harabasz_score(self.XW_consensus_, self.G_)
