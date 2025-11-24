from __future__ import annotations

from importlib.resources import files
from typing import Tuple, List

import numpy as np
from scipy import io
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer


def _load_mat_from_package(stem: str):
    """
    Load a .mat file bundled inside graphs.lmgec.datasets.

    Example:
        _load_mat_from_package("ACM") -> loads ACM.mat
    """
    path = files("graphs.lmgec.datasets") / f"{stem}.mat"
    with path.open("rb") as f:
        data = io.loadmat(f)
    return data


# -------- dataset loaders (ported from original utils.py) --------


def acm() -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    data = _load_mat_from_package("ACM")

    X = data["features"]
    A = data["PAP"]
    B = data["PLP"]

    Xs: List[np.ndarray] = [X.toarray()]
    As: List[np.ndarray] = [A.toarray(), B.toarray()]

    labels = data["label"].reshape(-1)
    return As, Xs, labels


def dblp() -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    data = _load_mat_from_package("DBLP")

    X = data["features"]
    A = data["net_APTPA"]
    B = data["net_APCPA"]
    C = data["net_APA"]

    Xs: List[np.ndarray] = [X.toarray()]
    As: List[np.ndarray] = [A.toarray(), B.toarray(), C.toarray()]

    labels = data["label"].reshape(-1)
    return As, Xs, labels


def imdb() -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    data = _load_mat_from_package("IMDB")

    X = data["features"]
    A = data["MAM"]
    B = data["MDM"]

    Xs: List[np.ndarray] = [X.toarray()]
    As: List[np.ndarray] = [A.toarray(), B.toarray()]

    labels = data["label"].reshape(-1)
    return As, Xs, labels


def photos() -> Tuple[List[sp.spmatrix], List[sp.spmatrix], np.ndarray]:
    data = _load_mat_from_package("Amazon_photos")

    X = data["features"]      # typically sparse
    A = data["adj"]           # adjacency
    labels = data["label"].reshape(-1)

    As = [A]
    Xs = [X, X @ X.T]

    return As, Xs, labels


def wiki() -> Tuple[List[sp.spmatrix], List[np.ndarray], np.ndarray]:
    data = _load_mat_from_package("wiki")

    X = data["fea"].toarray().astype(float)
    A = data["W"]
    labels = data["gnd"].reshape(-1)

    As = [A, kneighbors_graph(X, 5, metric="cosine")]
    Xs = [X, np.log2(1 + X)]

    return As, Xs, labels


def datagen(dataset: str):
    """
    Dispatch to a specific dataset loader, exactly like original utils.datagen().
    """
    dataset = dataset.lower()
    if dataset == "imdb":
        return imdb()
    if dataset == "dblp":
        return dblp()
    if dataset == "acm":
        return acm()
    if dataset == "photos":
        return photos()
    if dataset == "wiki":
        return wiki()
    raise ValueError(f"Unknown dataset: {dataset}")


def preprocess_dataset(adj, features, tf_idf: bool = False, beta: float = 1.0):
    """
    Port of utils.preprocess_dataset().

    adj is row-normalized after adding beta * I.
    features are either TF-IDF + l2, or just l2-normalized.
    """
    # This is intentionally very close to the original code
    # to minimize behavioral differences.
    adj = adj + beta * sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj)

    if tf_idf:
        features = TfidfTransformer(norm="l2").fit_transform(features)
    else:
        features = normalize(features, norm="l2")

    return adj, features


