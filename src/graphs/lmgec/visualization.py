import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import networkx as nx

class Visualizer:
    """
    visualizing Graph Clustering results.
    """

    @staticmethod
    def _align_labels(y_true, y_pred):
        """
        Helper to align unsupervised cluster labels with ground truth labels.
        1. Fixes 1-based indexing (1..6 -> 0..5).
        2. Uses Hungarian Algorithm to match predicted cluster ID to true class ID.
        """
        # Fix Indexing (Remove "Extra Class") 
        # If y_true starts at 1, shift it to 0
        if y_true.min() == 1 and y_pred.min() == 0:
            y_true = y_true - 1
            
        # Ensure they have the same set of unique labels to avoid dimension mismatch
        # (This handles the case where n_clusters != n_classes slightly better)
        unique_true = np.unique(y_true)
        n_classes = len(unique_true)
        
        # Hungarian Matching
        # Compute the cost matrix (negative confusion matrix)
        # We want to maximize the diagonal (matches), so we minimize the negative
        cm = confusion_matrix(y_true, y_pred)
        
        # If sizes don't match (e.g. 6 classes vs 7 clusters), crop to common size for matching
        r, c = cm.shape
        if r != c:
            # Fallback: Just return as is if dimensions are wildly different
            print(f"Warning: Label dimension mismatch ({r} vs {c}). Skipping alignment.")
            return y_true, y_pred

        # Find best permutation
        row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
        
        # Create a map: Predicted Label -> Best True Label
        # col_ind[i] is the column (prediction) that matches row_ind[i] (truth)
        reordering_map = {col: row for row, col in zip(row_ind, col_ind)}
        
        # Apply mapping to predictions
        y_pred_aligned = np.array([reordering_map.get(y, y) for y in y_pred])
        
        return y_true, y_pred_aligned

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None, match_labels=True):
        """
        Plots a heatmap of the confusion matrix.
        
        Args:
            match_labels (bool): If True, runs Hungarian algorithm to align 
                                 unsupervised clusters to the best matching true labels.
        """
        # Copy to avoid modifying original arrays outside
        y_t = y_true.copy()
        y_p = y_pred.copy()

        if match_labels:
            print("Aligning labels using Hungarian Algorithm...")
            y_t, y_p = Visualizer._align_labels(y_t, y_p)

        cm = confusion_matrix(y_t, y_p)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(title)
        plt.xlabel("Predicted Label (Aligned)")
        plt.ylabel("True Label")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    # Keep plot_embedding and plot_graph_structure as they were
    @staticmethod
    def plot_embedding(H, labels, method='tsne', title="Embedding Visualization", save_path=None):
        print(f"Running {method.upper()} reduction...")
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        else:
            reducer = PCA(n_components=2)
        H_2d = reducer.fit_transform(H)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(H_2d[:, 0], H_2d[:, 1], c=labels, cmap='viridis', s=10, alpha=0.6)
        plt.colorbar(scatter, label="Cluster Label")
        plt.title(title)
        plt.xlabel(f"{method.upper()} Dim 1")
        plt.ylabel(f"{method.upper()} Dim 2")
        plt.grid(True, linestyle='--', alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()

    @staticmethod
    def plot_graph_structure(adj, labels=None, max_nodes=500, title="Graph Structure"):
        import scipy.sparse as sp
        if adj.shape[0] > max_nodes:
            print(f"Graph too large ({adj.shape[0]} nodes). Plotting first {max_nodes} nodes only.")
            adj = adj[:max_nodes, :max_nodes]
            if labels is not None:
                labels = labels[:max_nodes]
        G = nx.from_scipy_sparse_array(adj) if sp.issparse(adj) else nx.from_numpy_array(adj)
        G.remove_edges_from(nx.selfloop_edges(G))
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, k=0.15, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=30, node_color=labels if labels is not None else 'blue', cmap='viridis', alpha=0.8)
        nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()