from __future__ import annotations

import numpy as np
from importlib.resources import files
from scipy import io 
from sklearn.neighbors import kneighbors_graph

def _load_mat_from_package(filename: str):
    """
    Load a .mat file that is bundled within the package.

    filename: e.g "ACM.mat"

    """

    path = files("graphs.lmgec.datasets") / filename
    # loadmat accepts file-like, so we open the resource
    with path.open("rb") as f:
        data = io.loadmat(f)
    return data

def acm():
    data = _load_mat_from_package("ACM.mat")
    X = data["features"]
    A = data["PAP"]
    B = data["PLP"]

    Xs = [X.toarray()]
    As = [A.toarray(), B.toarray()]

    labels = data["label"].reshape(-1)

    return As, Xs, labels

def dblp():
    data = _load_mat_from_package("DBLP.mat")

    X = data["features"]
    A = data["net_APTPA"]
    B = data["net_APCPA"]
    C = data["net_APA"]

    Xs = [X.toarray()]
    As = [A.toarray(), B.toarray(), C.toarray()]

    labels = data["label"].reshape(-1)

    return As, Xs, labels


def imdb():
    data = _load_mat_from_package("IMDB.mat")

    X = data["features"]
    A = data["MAM"]
    B = data["MDM"]

    Xs = [X.toarray()]
    As = [A.toarray(), B.toarray()]

    labels = data["label"].reshape(-1)

    return As, Xs, labels

def photos():
    data = _load_mat_from_package("Amazon_photos.mat")

    X = data["features"]
    A = data["adj"]
    labels = data["label"].reshape(-1)

    As = [A]
    Xs = [X, X @ X.T]

    return As, Xs, labels

def wiki():
    data = _load_mat_from_package("wiki.mat")

    X = data["fea"].toarray().astype(float)
    A = data["W"]
    labels = data["gnd"].reshape(-1)

    As = [A, kneighbors_graph(X, 5, metric="cosine")]
    Xs = [X, np.log2(1 + X)]

    return As, Xs, labels

