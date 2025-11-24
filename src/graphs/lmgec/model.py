"""
LMGEC model definition.

NumPy / scikit-learn implementation of the LMGEC (Linear Multi-view Graph
Embedding Clustering) algorithm, originally implemented in TensorFlow.

This module exposes:

- LMGEC: sklearn-style estimator with fit / fit_predict and accessors.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

try:
    # scikit-learn base classes (optional)
    from sklearn.base import BaseEstimator, ClusterMixin
except ImportError:  # fallback if sklearn isn't installed yet
    BaseEstimator = object  # type: ignore
    ClusterMixin = object  # type: ignore

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD


Array2D = np.ndarray


# ---------------------------------------------------------------------------
# Internal helpers: NumPy replacement for original TensorFlow code
# ---------------------------------------------------------------------------


def _init_W(
    X: Array2D,
    m: int,
    random_state: Optional[int] = None,
) -> Array2D:
    """
    Initialize W via truncated SVD on X.

    X : (n_samples, d)
    returns W : (d, m) with orthonormal-ish columns (approx).
    """
    svd = TruncatedSVD(n_components=m, random_state=random_state)
    svd.fit(X)
    # sklearn's components_ is (m, d) => transpose to (d, m)
    return svd.components_.T


def _init_G_F(
    XW: Array2D,
    k: int,
    random_state: Optional[int] = None,
) -> tuple[np.ndarray, Array2D]:
    """
    Initialize cluster labels G and centers F using KMeans on XW.

    XW : (n_samples, m)
    k  : number of clusters

    returns
    -------
    G : (n_samples,) integer cluster labels
    F : (k, m) cluster centers
    """
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    km.fit(XW)
    return km.labels_, km.cluster_centers_


def _update_W(
    X: Array2D,
    F: Array2D,
    G: np.ndarray,
) -> Array2D:
    """
    Update W via orthogonal Procrustes step, as in the original code.

    X : (n, d)
    F : (k, m)
    G : (n,) cluster labels

    Compute SVD of M = X^T @ F[G], then set W = U V^T.
    """
    # F[G] is (n, m): row i is center of cluster G[i]
    M = X.T @ F[G]  # (d, m)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt  # (d, m)


def _update_G(
    XW: Array2D,
    F: Array2D,
) -> np.ndarray:
    """
    Update G by assigning each point in XW to nearest center in F
    under mean squared distance.

    XW : (n, m)
    F  : (k, m)
    """
    # diff : (n, k, m)
    diff = XW[:, None, :] - F[None, :, :]
    dists = np.mean(diff ** 2, axis=2)  # (n, k)
    return np.argmin(dists, axis=1)     # (n,)


def _update_F(
    XW: Array2D,
    G: np.ndarray,
    k: int,
) -> Array2D:
    """
    Update F as cluster-wise means of XW.

    This mirrors tf.math.unsorted_segment_mean:
      - For each cluster j, F[j] is mean of points with label j;
      - If a cluster has no points, F[j] stays zero.

    XW : (n, m)
    G  : (n,)
    k  : number of clusters
    """
    n, m = XW.shape
    F = np.zeros((k, m), dtype=XW.dtype)
    for j in range(k):
        mask = (G == j)
        if np.any(mask):
            F[j] = XW[mask].mean(axis=0)
        # else: leave F[j] as zeros (unsorted_segment_mean behavior)
    return F


def _train_loop(
    Xs: Sequence[Array2D],
    F: Array2D,
    G: np.ndarray,
    alphas: np.ndarray,
    k: int,
    max_iter: int,
    tolerance: float,
) -> tuple[np.ndarray, Array2D, Array2D, np.ndarray]:
    """
    Main training loop, NumPy version of the original TensorFlow train_loop.

    Parameters
    ----------
    Xs : list of H_v matrices (one per view), each (n, d_v)
    F  : initial centers (k, m)
    G  : initial labels (n,)
    alphas : (V,) view weights
    k  : number of clusters
    max_iter : max number of iterations
    tolerance : convergence threshold on loss difference; if 0, no early stop

    Returns
    -------
    G            : final labels (n,)
    F            : final centers (k, m)
    XW_consensus : final consensus embedding (n, m)
    loss_history : array of losses, shape (t,)
    """
    n_views = len(Xs)
    n = Xs[0].shape[0]
    m = F.shape[1]

    loss_history: List[float] = []
    prev_loss = np.inf

    for _ in range(max_iter):
        loss = 0.0
        XW_consensus = np.zeros((n, m), dtype=float)

        for v in range(n_views):
            Xv = Xs[v]  # (n, d_v)

            # Update W_v
            Wv = _update_W(Xv, F, G)       # (d_v, m)
            XWv = Xv @ Wv                  # (n, m)

            # Accumulate weighted consensus embedding
            XW_consensus += alphas[v] * XWv

            # Reconstruction error for this view:
            # F @ Wv^T : (k, m) @ (m, d_v) -> (k, d_v)
            # gather by G -> (n, d_v)
            recon = (F @ Wv.T)[G]
            diff = Xv - recon
            loss_v = np.linalg.norm(diff)
            loss += alphas[v] * loss_v

        # Update G and F using consensus embedding
        G = _update_G(XW_consensus, F)
        F = _update_F(XW_consensus, G, k)

        loss_history.append(loss)
        if tolerance is not None and tolerance > 0.0:
            if abs(prev_loss - loss) < tolerance:
                break
        prev_loss = loss

    return G, F, XW_consensus, np.array(loss_history, dtype=float)


def _lmgec_core(
    Xs: Sequence[Array2D],
    k: int,
    m: int,
    temperature: float,
    max_iter: int,
    tolerance: float,
    random_state: Optional[int] = None,
) -> tuple[np.ndarray, Array2D, Array2D, np.ndarray]:
    """
    Core LMGEC function. NumPy/scikit-learn analog of the original lmgec().

    Parameters
    ----------
    Xs : list of H_v matrices (n, d_v) after propagation + scaling.
    k  : number of clusters
    m  : embedding dimension
    temperature : temperature for alpha softmax
    max_iter : training iterations
    tolerance : convergence threshold
    random_state : random seed (used for SVD and KMeans)

    Returns
    -------
    G            : labels (n,)
    F            : centers (k, m)
    XW_consensus : consensus embedding (n, m)
    loss_history : (t,)
    """
    n_views = len(Xs)
    if n_views == 0:
        raise ValueError("Xs must contain at least one view.")

    # Initialize view weights and consensus embedding
    alphas = np.zeros(n_views, dtype=float)
    XW_consensus: Optional[Array2D] = None

    for v, Xv in enumerate(Xs):
        Xv = np.asarray(Xv, dtype=float)

        # 1) init W_v and per-view embedding
        Wv = _init_W(Xv, m, random_state=random_state)  # (d_v, m)
        XWv = Xv @ Wv                                   # (n, m)

        # 2) init G_v, F_v via k-means on XWv
        Gv, Fv = _init_G_F(XWv, k, random_state=random_state)

        # 3) inertia and view weight alpha_v
        inertia = np.linalg.norm(XWv - Fv[Gv])
        alphas[v] = np.exp(-inertia / temperature)

        # 4) accumulate weighted consensus embedding
        if XW_consensus is None:
            XW_consensus = alphas[v] * XWv
        else:
            XW_consensus += alphas[v] * XWv

    if XW_consensus is None:
        raise RuntimeError("Failed to initialize XW_consensus.")

    alpha_sum = alphas.sum()
    if alpha_sum == 0.0:
        # fallback: if all alphas collapsed to zero (degenerate case)
        alphas[:] = 1.0
        alpha_sum = float(n_views)
    XW_consensus /= alpha_sum

    # Initial global G, F from consensus
    G, F = _init_G_F(XW_consensus, k, random_state=random_state)

    # Training loop
    G, F, XW_consensus, loss_history = _train_loop(
        Xs, F, G, alphas, k, max_iter, tolerance
    )

    return G, F, XW_consensus, loss_history


def _compute_view_embeddings(
    Xs: Sequence[Array2D],
    F: Array2D,
    G: np.ndarray,
) -> List[Array2D]:
    """
    Compute per-view embeddings X_v W_v using the final F, G.

    Not in the original implementation, but useful to expose.
    """
    embeddings: List[Array2D] = []
    for Xv in Xs:
        Xv = np.asarray(Xv, dtype=float)
        Wv = _update_W(Xv, F, G)  # (d_v, m)
        embeddings.append(Xv @ Wv)
    return embeddings


# ---------------------------------------------------------------------------
# Public estimator class
# ---------------------------------------------------------------------------


class LMGEC(BaseEstimator, ClusterMixin):
    """
    LMGEC multi-view graph clustering.

    This class assumes you pass in a list of preprocessed view matrices Xs,
    where each X_v corresponds to the H_v used in the original lmgec code:

        Original run.py:
            Hs = []
            for each view:
                features = S @ X
                x = StandardScaler(with_std=False).fit_transform(features)
                Hs.append(x)

            Z, F, XW_consensus, losses = lmgec(Hs, k, k+1, ...)

        Here:
            model = LMGEC(n_clusters=k, n_components=k+1, tau=temperature, ...)
            model.fit(Hs)

    Parameters
    ----------
    n_clusters : int
        Number of clusters (k).
    n_components : int, optional
        Embedding dimension (m). If None, defaults to k + 1.
    beta : float, default=1.0
        Kept for API compatibility; used in preprocessing layer, not in core.
    tau : float, default=1.0
        Temperature parameter for the view-weight softmax (same as "temperature"
        in the original code).
    max_iter : int, default=30
        Maximum number of BCD iterations.
    tol : float, default=0.0
        Tolerance on loss difference for early stopping.
        If 0.0, runs for exactly max_iter iterations.
    random_state : int, optional
        Random seed.
    use_sparse : bool, default=True
        Placeholder for future sparse support (not used directly here).
    """

    def __init__(
        self,
        n_clusters: int,
        n_components: Optional[int] = None,
        beta: float = 1.0,
        tau: float = 1.0,
        max_iter: int = 30,
        tol: float = 0.0,
        random_state: Optional[int] = None,
        use_sparse: bool = True,
    ) -> None:
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.beta = beta
        self.tau = tau
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.use_sparse = use_sparse

        # Attributes set after fitting
        self.labels_: Optional[np.ndarray] = None
        self.embedding_: Optional[Array2D] = None
        self.membership_: Optional[Array2D] = None
        self.cluster_centers_: Optional[Array2D] = None
        self.view_embeddings_: Optional[List[Array2D]] = None
        self.loss_history_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Core sklearn-like API
    # ------------------------------------------------------------------

    def fit(
        self,
        Xs: Sequence[Array2D],
        y: Optional[np.ndarray] = None,
    ) -> "LMGEC":
        """
        Fit LMGEC on a list of preprocessed view matrices Xs.

        Parameters
        ----------
        Xs : sequence of array-like, each shape (n_samples, d_v)
            The H_v matrices for each view (after adjacency propagation and
            optional scaling), as in the original lmgec implementation.
        y : ignored
            Present for API compatibility.

        Returns
        -------
        self : LMGEC
            Fitted estimator.
        """
        if not Xs:
            raise ValueError("Xs must contain at least one view.")

        # Convert to float arrays and check consistent number of samples
        Xs_arr: List[Array2D] = [np.asarray(X, dtype=float) for X in Xs]
        n_samples = Xs_arr[0].shape[0]
        for X in Xs_arr:
            if X.shape[0] != n_samples:
                raise ValueError("All views must have the same number of samples.")

        k = self.n_clusters
        m = self.n_components if self.n_components is not None else (k + 1)

        # Run core algorithm
        G, F, Z, loss_history = _lmgec_core(
            Xs_arr,
            k=k,
            m=m,
            temperature=self.tau,
            max_iter=self.max_iter,
            tolerance=self.tol,
            random_state=self.random_state,
        )

        # Store results
        self.labels_ = G
        self.embedding_ = Z
        self.cluster_centers_ = F
        self.loss_history_ = loss_history
        self.membership_ = np.eye(k, dtype=int)[G]
        self.view_embeddings_ = _compute_view_embeddings(Xs_arr, F, G)

        return self

    def fit_predict(
        self,
        Xs: Sequence[Array2D],
        y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fit LMGEC and return cluster labels.

        Parameters
        ----------
        Xs : sequence of array-like
            Preprocessed view matrices.
        y : ignored

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        self.fit(Xs, y)
        if self.labels_ is None:
            raise RuntimeError("fit() did not set labels_.")
        return self.labels_

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_node_embeddings(self) -> Array2D:
        """Return the consensus node embeddings Z (shape: n_samples × m)."""
        if self.embedding_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.embedding_

    def get_view_embeddings(self) -> List[Array2D]:
        """Return the list of per-view embeddings X_v W_v."""
        if self.view_embeddings_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.view_embeddings_

    def get_membership_matrix(self) -> Array2D:
        """Return the cluster membership matrix G (one-hot, shape: n_samples × k)."""
        if self.membership_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.membership_

    def get_cluster_centers(self) -> Array2D:
        """Return the cluster center matrix F (shape: k × m)."""
        if self.cluster_centers_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.cluster_centers_
