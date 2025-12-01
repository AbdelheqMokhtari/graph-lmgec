from __future__ import annotations


import scipy.sparse as sp
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer


def preprocess_dataset(adj, features, tf_idf=False, beta=1):
    adj = adj + beta * sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj)

    if tf_idf:
        features = TfidfTransformer(norm="l2").fit_transform(features)
    else: 
        features = normalize(features, norm="l2")

    return adj, features

def lmgec_test():
    print("loading successfully")