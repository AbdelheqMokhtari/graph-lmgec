"""
graphs

A Collection of graph-related algorithms 

Subpackages
----------------
- graphs.lmgec: Implementation of the LMGeC algorithm for graph clustering and representation learning.
"""

from . import lmgec

__all__ = ['lmgec']

from .graph_construction import build_knn_graph, build_gaussian_graph, generate_multiview_graphs