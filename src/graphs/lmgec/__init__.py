"""
graphs.lmgec

Implementation of the LMGEC algorithm for graph clustering and representation learning.

Main entry point:

"""
import importlib.metadata

try:
    # This must match the 'name' in your pyproject.toml
    __version__ = importlib.metadata.version("graph-lmgec")
except importlib.metadata.PackageNotFoundError:
    # Fallback if the package is not installed (e.g., local dev without -e)
    __version__ = "unknown"

from .model import LMGEC
from .metrics import (
    clustering_accuracy,
    clustering_f1_score,
    evaluate_clustering,
)
from .utils import (
    get_propagated_features
)

__all__ = [
    "LMGEC",
    "clustering_accuracy",
    "clustering_f1_score",
    "evaluate_clustering",
    "preprocess_dataset",
    "lmgec_test"
]
