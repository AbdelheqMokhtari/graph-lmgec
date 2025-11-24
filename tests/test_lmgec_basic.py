"""
Basic smoke tests for the LMGEC skeleton.

Right now this just checks that the class can be imported and instantiated.
"""

from graphs.lmgec import LMGEC
from graphs.lmgec.datasets import acm


def test_lmgec_on_acm_smoke():
    As, Xs, labels = acm()
    # Build views list: (A_v, X_v)
    views = list(zip(As, [Xs[0]] * len(As)))

    model = LMGEC(n_clusters=len(set(labels)))
    labels_pred = model.fit_predict(views)

    assert labels_pred.shape == labels.shape
