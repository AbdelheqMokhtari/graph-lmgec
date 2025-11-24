"""
Example script to test LMGEC on 4 benchmark datasets:
ACM, DBLP, IMDB, Amazon Photos.

It:
- loads each dataset from graphs.lmgec.datasets
- builds multi-view features (Hs) similarly to the original code
- runs LMGEC
- computes ACC, F1, NMI, ARI, and final loss
- prints results per dataset
"""

from __future__ import annotations

import numpy as np
from itertools import product
from time import time
from typing import Callable, Tuple, List

from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp

from graphs.lmgec import LMGEC
from graphs.lmgec.datasets import acm, dblp, imdb, photos
from graphs.lmgec.metrics import evaluate_clustering


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _row_normalized_adj(A, beta: float = 1.0):
    """
    Simple adjacency preprocessing:
    S = D̃^{-1} (A + beta * I), using row-normalization.

    Works for dense and sparse matrices.
    """
    if sp.issparse(A):
        A = A.tocsr()
        n = A.shape[0]
        A_tilde = A + beta * sp.eye(n, format="csr")
        d = np.array(A_tilde.sum(axis=1)).ravel()
        d[d == 0] = 1.0
        inv_d = 1.0 / d
        D_inv = sp.diags(inv_d)
        S = D_inv @ A_tilde
        return S
    else:
        A = np.asarray(A, dtype=float)
        n = A.shape[0]
        A_tilde = A + beta * np.eye(n)
        d = A_tilde.sum(axis=1)
        d[d == 0] = 1.0
        S = A_tilde / d[:, None]
        return S


def _build_views_Hs(
    As: List,
    Xs: List,
    beta: float = 1.0,
    standardize: bool = True,
) -> List[np.ndarray]:
    """
    Mimic what the original run.py does:

    views = list(product(As, Xs))
    for each (A, X):
        S, features = preprocess_dataset(...)
        H = S @ features
        Hs.append(StandardScaler(with_std=False).fit_transform(H))

    Here we implement a simple version:
    - S = row-normalized (A + beta I)
    - features = X as given
    - H = S @ X
    """
    from itertools import product

    Hs: List[np.ndarray] = []

    for A, X in product(As, Xs):
        S = _row_normalized_adj(A, beta=beta)

        # convert X to dense if needed
        if sp.issparse(X):
            X = X.toarray()
        else:
            X = np.asarray(X, dtype=float)

        # S @ X can be sparse or dense depending on S
        HX = S @ X
        if sp.issparse(HX):
            HX = HX.toarray()

        if standardize:
            HX = StandardScaler(with_std=False).fit_transform(HX)

        Hs.append(HX.astype(float))

    return Hs


def _run_single_dataset(
    name: str,
    loader: Callable[[], Tuple[List, List, np.ndarray]],
    beta: float = 1.0,
    tau: float = 1.0,
    max_iter: int = 10,
    random_state: int = 0,
):
    """
    Load a dataset, build Hs, run LMGEC, and print metrics.
    """
    print(f"\n==================== {name.upper()} ====================")

    # 1. Load dataset
    As, Xs, labels = loader()
    labels = np.asarray(labels).ravel()
    k = len(np.unique(labels))
    print(f"Number of nodes: {len(labels)}, number of clusters: {k}")

    # 2. Build multi-view features Hs (one per (A, X) combination)
    Hs = _build_views_Hs(As, Xs, beta=beta, standardize=True)
    print(f"Number of views (Hs): {len(Hs)}")

    # 3. Run LMGEC
    model = LMGEC(
        n_clusters=k,
        n_components=k + 1,
        tau=tau,
        max_iter=max_iter,
        tol=0.0,
        random_state=random_state,
    )

    t0 = time()
    y_pred = model.fit_predict(Hs)
    elapsed = time() - t0

    # 4. Metrics
    final_loss = (
        model.loss_history_[-1] if model.loss_history_ is not None else None
    )
    metrics = evaluate_clustering(labels, y_pred, loss=final_loss)

    print(f"Time: {elapsed:.4f} s")
    print(f"ACC:  {metrics['acc']:.4f}")
    print(f"F1:   {metrics['f1']:.4f}")
    print(f"NMI:  {metrics['nmi']:.4f}")
    print(f"ARI:  {metrics['ari']:.4f}")
    if "loss" in metrics:
        print(f"Loss: {metrics['loss']:.4f}")

    return metrics


def main():
    """
    Run LMGEC on 4 datasets and print metrics.
    """
    beta = 2.0         # self-loop weight in adjacency normalization
    tau = 10.0          # temperature for view-weight softmax
    max_iter = 3
    random_state = 0

    datasets = [
        ("acm", acm),
        ("dblp", dblp),
        ("imdb", imdb),
        ("photos", photos),
    ]

    results = {}
    for name, loader in datasets:
        metrics = _run_single_dataset(
            name,
            loader,
            beta=beta,
            tau=tau,
            max_iter=max_iter,
            random_state=random_state,
        )
        results[name] = metrics

    print("\n==================== SUMMARY ====================")
    for name, m in results.items():
        print(
            f"{name.upper()}: "
            f"ACC={m['acc']:.4f}, "
            f"F1={m['f1']:.4f}, "
            f"NMI={m['nmi']:.4f}, "
            f"ARI={m['ari']:.4f}, "
            f"Loss={m.get('loss', float('nan')):.4f}"
        )


if __name__ == "__main__":
    main()
