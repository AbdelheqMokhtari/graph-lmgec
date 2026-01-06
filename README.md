# graph-lmgec

Multi-view graph clustering and representation learning in Python. This package provides a reference implementation of LMGEC (Linear Multi-view Graph Embedding Clustering) together with utilities for graph construction, preprocessing, embedding visualization, and clustering metrics.

The code is organized as a regular Python package under `src/graphs/lmgec`.


## Features

- LMGEC algorithm implemented with NumPy/Scikit-learn plus a small TensorFlow loop for the alternating updates.
- Graph construction helpers and examples: build k-NN graphs from feature matrices or construct graphs from edge lists with NetworkX, then normalize for learning.
- Preprocessing utilities: adjacency normalization and TF–IDF/ℓ2 feature normalization.
- Embedding and graph visualization examples: plot low-dimensional embeddings colored by clusters; draw graphs with node colors.
- Cluster evaluation utilities: Hungarian-aligned clustering accuracy and F1.


## Installation

Requires Python 3.8+.

Install the package in editable mode:

```bash
pip install -e .
```

Dependencies are declared in `pyproject.toml` and include NumPy, SciPy, scikit-learn, NetworkX, and pandas. The core LMGEC training loop currently uses TensorFlow; please install it explicitly if you plan to train the model:

```bash
# CPU-only TensorFlow (example)
pip install "tensorflow>=2.9"
```


## Package structure

```
LICENSE                 # MIT license
pyproject.toml          # Project metadata and dependencies
README.md               # This file

src/
	graphs/
		__init__.py         # Namespace package init
		lmgec/
			__init__.py       # Public exports
			model.py          # LMGEC implementation (fit loop and updates)
			metrics.py        # Clustering metrics (ACC, F1 via Hungarian alignment)
			utils.py          # Preprocessing helpers (e.g., adjacency normalization)
			graph_construction.py 
            visualization.py
tests/
	__init__.py           # Marks tests as a package
	run.py                # Example end-to-end run script
```


## Quickstart

Bring your own data: start from a feature matrix X (n_samples × d) and construct one or more graphs from it. Below we build two k-NN graphs with different k values to simulate two views, then run LMGEC.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp
import networkx as nx

from graphs.lmgec import preprocess_dataset  # from utils
from graphs.lmgec import LMGEC

# 1) Feature matrix (example) — replace with your own
n, d = 1000, 128
X = np.random.RandomState(0).randn(n, d).astype(float)

# 2) Build graphs (two views) from features
A1 = kneighbors_graph(X, n_neighbors=10, metric="cosine")
A2 = kneighbors_graph(X, n_neighbors=25, metric="cosine")
As = [A1, A2]
Xs = [X, X]  # same features for both views in this example

# 3) Preprocess each view and build H_v = S_v @ X_v
views = []
for A, Xv in zip(As, Xs):
  S, Xn = preprocess_dataset(A, Xv, tf_idf=False, beta=1.0)
  if not isinstance(Xn, np.ndarray):
    Xn = Xn.toarray()
  if type(S) == np.matrix:
    S = np.asarray(S)
  H = S @ Xn
  H = StandardScaler(with_std=False).fit_transform(H)
  views.append(H)

# 4) Fit LMGEC (choose k)
k = 10
Z, F, XW_consensus, losses = LMGEC(k, k+1, 10.0, 10).fit(views)

print("labels shape:", Z.shape)
print("final loss:", losses[-1])
```

You can also run the prebuilt script:

```bash
python tests/run.py
```


## API overview

Public objects are exported in `graphs.lmgec.__init__`. The package groups functionality into four areas: model, utils (preprocessing), metrics, and recipes for graph construction and visualization.

### 1) Model

- `LMGEC(n_clusters, n_components, temperature, max_iter)`
	- Fit: `Z, F, XW_consensus, loss_history = LMGEC(...).fit(Hs)`
	- Inputs:
		- `Hs`: list of V arrays (each `(n_samples, d_v)`), typically built as `H_v = S_v @ X_v` per view.
	- Outputs:
		- `Z` (labels): `(n_samples,)` integers in `[0, n_clusters-1]`.
		- `F` (centroids): `(n_clusters, n_components)`.
		- `XW_consensus`: `(n_samples, n_components)` consensus embedding.
		- `loss_history`: 1-D array of length `max_iter`.
	- Note: a scikit-learn-style `fit_predict` is not yet exposed; use `fit(...)` and take `Z`.

### 2) Utils (preprocessing)

- `preprocess_dataset(adj, features, tf_idf=False, beta=1.0) -> (norm_adj, norm_features)`
	- `adj`: `(n, n)` SciPy sparse or ndarray; `features`: `(n, d)` sparse or ndarray.
	- Adds `beta * I` to `adj`, then row-normalizes it.
	- If `tf_idf=True`, applies TF–IDF (with ℓ2 norm) to `features`; otherwise ℓ2-normalizes features.
	- Returns: `norm_adj` (sparse or ndarray) and `norm_features` (sparse or ndarray). Convert to dense for TensorFlow if needed.

- `lmgec_test()`
	- Simple smoke helper that prints a message (useful during setup).

### 3) Metrics

Permutation-invariant clustering scores (internally align labels via the Hungarian algorithm):

- `clustering_accuracy(y_true, y_pred) -> float`
	- Accuracy computed from the reordered confusion matrix.
- `clustering_f1_score(y_true, y_pred, **kwargs) -> float`
	- F1 computed on a pseudo-expanded label sequence; accepts scikit-learn F1 kwargs (e.g., `average="macro"`).
- `evaluate_clustering(y_true, y_pred, average="macro") -> dict`
	- Returns `{"acc": acc, "f1": f1}` as floats.

### 4) Graph construction (GraphBuilder)

High-level helper to construct adjacency matrices from feature matrices using common similarity rules.

Import:

```python
from graphs.lmgec.graph_construction import GraphBuilder
```

API:
- `GraphBuilder.build(Xs, method='knn', **kwargs) -> list[sp.csr_matrix]`
  - `Xs`: list of feature matrices `[X1, X2, ...]` each `(n, d_v)`.
  - `method`: `'knn'`, `'rbf'` (alias `'gaussian'`), or `'cosine'`.
  - returns a list of CSR sparse adjacency matrices `[A1, A2, ...]`.

- `GraphBuilder.knn(X, k=10, metric='euclidean', mode='connectivity', include_self=False)`
  - Builds a k-NN graph via scikit-learn. `mode='connectivity'` gives 0/1; `'distance'` yields weighted edges. Optionally include self-loops.

- `GraphBuilder.rbf(X, sigma=1.0, threshold=None, k=None)`
  - Dense RBF kernel `A_ij = exp(-||x_i-x_j||^2/(2 sigma^2))`, then sparsifies by `threshold` and/or keeps top-`k` per row. Returns CSR.

- `GraphBuilder.cosine(X, k=None, threshold=None)`
  - Dense cosine similarity (negatives zeroed), with optional threshold/top-`k` sparsification. Returns CSR.

Example — two views from a single feature matrix:

```python
Xs = [X, X]
As = GraphBuilder.build(Xs, method='knn', k=15, metric='cosine', mode='connectivity')
```

Example — Gaussian kernels with sparsification:

```python
As = GraphBuilder.build([X1, X2], method='rbf', sigma=0.7, k=20)  # keep top-20 per row
```

Then normalize and build `H_v = S_v @ X_v` per view:

```python
from graphs.lmgec import preprocess_dataset
Hs = []
for A, Xv in zip(As, Xs):
  S, Xn = preprocess_dataset(A, Xv, tf_idf=False, beta=1.0)
  if hasattr(Xn, 'toarray'):
    Xn = Xn.toarray()
  Hs.append(S @ Xn)
```

Notes:
- RBF and cosine first build a dense similarity matrix before sparsifying; prefer `knn` for large `n` to avoid `O(n^2)` memory.
- Outputs are `scipy.sparse.csr_matrix`; symmetrize if required for your pipeline.

Low-level recipes (alternative to GraphBuilder):

- k-NN graph from features:
	```python
	from sklearn.neighbors import kneighbors_graph
	A = kneighbors_graph(X, n_neighbors=15, metric="cosine")  # (n, n) sparse
	```

- Edge list to adjacency with NetworkX:
	```python
	import pandas as pd, networkx as nx
	df = pd.read_csv("edges.csv")  # columns: source, target
	G = nx.from_pandas_edgelist(df, "source", "target")
	A = nx.to_scipy_sparse_array(G, dtype=float)
	```

- Combine multiple graphs (views):
	```python
	A_comb = 0.5 * A_view1 + 0.5 * A_view2
	# Optional: symmetrize if needed
	# A_comb = 0.5 * (A_comb + A_comb.T)
	```

### 5) Visualization (Visualizer)

High-level plotting utilities live in `graphs.lmgec.visualization.Visualizer`. They help you:

- Align and visualize confusion matrices for clustering results.
- Project embeddings to 2D with t-SNE or PCA and color by cluster labels.
- Draw graph structures for inspection (small graphs recommended).

Import:

```python
from graphs.lmgec.visualization import Visualizer
```

APIs:

- `Visualizer.plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None, match_labels=True)`
	- Computes a confusion matrix and draws a Seaborn heatmap.
	- If `match_labels=True` (default), internally aligns labels via Hungarian assignment for best class–cluster matching. It also fixes 1-based label indexing (1..C -> 0..C-1) when needed.
	- `save_path`: optional file path to save the figure (PNG, etc.).
	- Example:
		```python
		Visualizer.plot_confusion_matrix(labels_true, Z, title="LMGEC vs Ground Truth", match_labels=True)
		```

- `Visualizer.plot_embedding(H, labels, method='tsne', title="Embedding Visualization", save_path=None)`
	- Reduces an embedding `H` to 2D and scatters points colored by `labels`.
	- `method`: `'tsne'` (default, TSNE with PCA init and auto LR) or `'pca'` (fast linear projection).
	- `save_path`: optional path to save the figure.
	- Example (use the model’s consensus embedding):
		```python
		Visualizer.plot_embedding(XW_consensus, Z, method='pca', title='Consensus (PCA)')
		```

- `Visualizer.plot_graph_structure(adj, labels=None, max_nodes=500, title="Graph Structure")`
	- Draws a spring-layout visualization of a graph from an adjacency matrix `adj` (SciPy sparse or dense ndarray). Self-loops are removed for readability.
	- If `adj` has more than `max_nodes` rows, only the top-left `max_nodes × max_nodes` block is plotted, and `labels` are truncated accordingly.
	- Example:
		```python
		Visualizer.plot_graph_structure(As[0], labels=Z, max_nodes=400, title='View 1 (spring layout)')
		```

Notes:
- These utilities depend on `matplotlib` and `seaborn` (for the heatmap) and `networkx` for graph drawing. Install them if they’re not present: `pip install matplotlib seaborn networkx`.
- t-SNE is stochastic and O(n²) in memory/time. For quick overviews, prefer `method='pca'` on large datasets.


## Algorithm (LMGEC) — intuition and updates

LMGEC seeks a shared low-dimensional representation across multiple views and performs clustering jointly. At a high level, for each view v with features `X_v` we learn an orthonormal projection `W_v` into a common `m`-dimensional space, build a weighted consensus embedding, and alternate cluster assignment and centroid updates.

Let `k` be the number of clusters, `m` the embedding dimension, `F ∈ R^{k×m}` the centroids, `G ∈ {0..k-1}^n` the assignments, and `W_v ∈ R^{d_v×m}`. The per-view projected features are `X_v W_v`. The consensus is a weighted sum across views, with weights `α_v` reflecting view reliability:

$$XW_{consensus} = \sum_v \alpha_v (X_v W_v).$$

We alternate the following updates:

- Update `W_v` (orthogonal Procrustes via SVD):
	$$W_v \leftarrow \operatorname*{argmin}_{W^T W = I} \|X_v - F_{G} W^T\|_F^2 \;\Rightarrow\; W_v = U V^T,$$
	where `U Σ V^T` is the SVD of `X_v^T (F_G)` and `F_G` stacks the centroid of each sample according to its current label.
- Update assignments `G` by nearest centroid in the consensus embedding:
	$$G_i \leftarrow \arg\min_c \; \| (XW_{consensus})_i - F_c \|_2^2.$$
- Update centroids `F` as cluster means:
	$$F_c \leftarrow \frac{1}{|\{i:G_i=c\}|} \sum_{i:G_i=c} (XW_{consensus})_i.$$

View weights `α_v` are initialized from a soft evidence score using KMeans inertia on each view’s initial embedding and a temperature parameter `τ`:

$$\alpha_v \propto \exp\Big(-\frac{\text{inertia}_v}{\tau}\Big).$$

Implementation details:

- Centroid update uses a segment-wise mean; assignment uses squared Euclidean distances.
- `W_v` update uses SVD from TensorFlow linear algebra.
- The outer loop runs for `max_iter` iterations and records a simple loss proxy per iteration.



## Tips and troubleshooting

- TensorFlow not installed: install `tensorflow>=2.9` for CPU or an appropriate GPU build.
- Visualization: install `matplotlib` for plotting (`pip install matplotlib`).


## License

MIT License — see `LICENSE`.



