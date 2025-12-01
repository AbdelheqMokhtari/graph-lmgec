"""
LMGEC model definition.

NumPy / scikit-learn implementation of the LMGEC (Linear Multi-view Graph
Embedding Clustering) algorithm, originally implemented in TensorFlow.

This module exposes:

- LMGEC: sklearn-style estimator with fit / fit_predict and accessors.
"""

import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import numpy as np

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

def _init_G_F(XW, k):
    km = KMeans(n_clusters=k).fit(XW)
    G = km.labels_
    F = km.cluster_centers_
    return G, F

def _init_W(X, n_components):
    svd = TruncatedSVD(n_components=n_components).fit(X)
    W = svd.components_.T
    return W

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


class LMGEC():

    def __init__(
        self,
        n_clusters:int,
        m: float,
        tempertaure:float,
        max_iter: int = 30,
    ) -> None:
        self.n_clusters = n_clusters
        self.m = m
        self.temperature = tempertaure
        self.max_iter = max_iter

    def fit(self, Xs):
        n_views = len(Xs)

        # inital G and F 
        alphas = np.zeros(n_views)

        XW_consensus = 0
        for v in range(n_views):
            Wv = _init_W(Xs[v], self.m)
            XWv = Xs[v]@Wv
            Gv, Fv = _init_G_F(XWv, self.n_clusters)
            intertia = np.linalg.norm(XWv - Fv[Gv])
            alphas[v] = np.exp(-intertia/self.temperature)
            XW_consensus += alphas[v] * XWv
        WX_consensus = XW_consensus / alphas.sum()
        G, F =_init_G_F(XW_consensus, self.n_clusters)

        G, F, XW_consensus, loss_history = _train_loop(Xs, F, G, alphas, self.n_clusters, self.max_iter)
        
        return G, F, XW_consensus, loss_history







