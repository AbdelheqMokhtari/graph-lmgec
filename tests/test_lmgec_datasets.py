from __future__ import annotations

import numpy as np
from itertools import product
from time import time

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.preprocessing import StandardScaler

from graphs.lmgec import (
    LMGEC,
    clustering_accuracy,
    clustering_f1_score,
)
from graphs.lmgec.datasets import (
    datagen,
    preprocess_dataset,
)


def run_single_dataset(
    dataset: str,
    temperature: float = 1.0,
    beta: float = 1.0,
    max_iter: int = 10,
    tol: float = 0.0,
    runs: int = 1,
    random_state: int = 0,
):
    """
    Reproduce the original run.py pipeline for a single dataset,
    using the LMGEC class instead of the TensorFlow function.
    """
    print(f"\n----------------- {dataset} -----------------")

    # Load dataset
    As, Xs, labels = datagen(dataset)
    labels = np.asarray(labels).reshape(-1)
    k = len(np.unique(labels))

    # All combinations of topology and features, like original code
    views = list(product(As, Xs))

    # Preprocess each view as in utils.preprocess_dataset()
    for v in range(len(views)):
        A, X = views[v]
        tf_idf = dataset.lower() in ["acm", "dblp", "imdb", "photos"]
        norm_adj, features = preprocess_dataset(A, X, tf_idf=tf_idf, beta=beta)

        if not isinstance(features, np.ndarray):
            features = features.toarray()

        # Original code handles np.matrix explicitly; we do the same
        if type(norm_adj) == np.matrix:
            norm_adj = np.asarray(norm_adj)

        views[v] = (norm_adj, features)

    metrics = {
        "acc": [],
        "nmi": [],
        "ari": [],
        "f1": [],
        "loss": [],
        "time": [],
    }

    for run in range(runs):
        t0 = time()

        # Build Hs exactly as in run.py: H = S @ X, then StandardScaler
        Hs = []
        for S, X in views:
            features = S @ X
            x = StandardScaler(with_std=False).fit_transform(features)
            Hs.append(x)

        # Run our NumPy implementation of LMGEC
        model = LMGEC(
            n_clusters=k,
            n_components=k + 1,
            tau=temperature,
            max_iter=max_iter,
            tol=tol,
            random_state=None if random_state is None else random_state + run,
        )

        Z = model.fit_predict(Hs)  # Z corresponds to G (cluster labels)
        elapsed = time() - t0

        final_loss = (
            model.loss_history_[-1]
            if model.loss_history_ is not None and len(model.loss_history_) > 0
            else np.nan
        )

        metrics["time"].append(elapsed)
        metrics["acc"].append(clustering_accuracy(labels, Z))
        metrics["nmi"].append(nmi(labels, Z))
        metrics["ari"].append(ari(labels, Z))
        metrics["f1"].append(
            clustering_f1_score(labels, Z, average="macro")
        )
        metrics["loss"].append(final_loss)

    results_mean = {k: float(np.mean(v)) for k, v in metrics.items()}
    results_std = {k: float(np.std(v)) for k, v in metrics.items()}

    print(
        f"ACC={results_mean['acc']:.4f}±{results_std['acc']:.4f}, "
        f"F1={results_mean['f1']:.4f}±{results_std['f1']:.4f}, "
        f"NMI={results_mean['nmi']:.4f}±{results_std['nmi']:.4f}, "
        f"ARI={results_mean['ari']:.4f}±{results_std['ari']:.4f}"
    )

    return {"mean": results_mean, "std": results_std}


def main():
    # You can change this list to run specific datasets
    datasets = ["acm", "dblp", "imdb", "photos", "wiki"]

    all_results = {}
    for ds in datasets:
        res = run_single_dataset(
            dataset=ds,
            temperature=1.0,
            beta=1.0,
            max_iter=3,
            tol=0.0,
            runs=1,
            random_state=42,
        )
        all_results[ds] = res

    print("\n==================== SUMMARY ====================")
    for ds, res in all_results.items():
        m = res["mean"]
        print(
            f"{ds.upper()}: "
            f"ACC={m['acc']:.4f}, "
            f"F1={m['f1']:.4f}, "
            f"NMI={m['nmi']:.4f}, "
            f"ARI={m['ari']:.4f}, "
            f"Loss={m['loss']:.4f}, "
            f"Time={m['time']:.4f}s"
        )


if __name__ == "__main__":
    main()
