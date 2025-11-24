"""
graphs.lmgec

Implementation of the LMGEC algorithm for graph clustering and representation learning.

Main entry point:

"""

from .model import LMGEC
from .metrics import (
    clustering_accuracy,
    clustering_f1_score,
    evaluate_clustering,
)

__all__ = [
    "LMGEC",
    "clustering_accuracy",
    "clustering_f1_score",
    "evaluate_clustering",
]
