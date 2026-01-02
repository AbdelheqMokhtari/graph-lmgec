import numpy as np
import scipy.io as io
from pathlib import Path
from itertools import product 
from graphs.lmgec import LMGEC

def load_data(file_path="datasets/ACM.mat"):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.absolute()}")

    with path.open("rb") as f:
        data = io.loadmat(f)

    # 1. Load Raw Components
    # We keep them separate initially
    X_raw = data["features"].toarray()  # Single feature matrix
    A_raw = data["PAP"].toarray()      # Graph 1
    B_raw = data["PLP"].toarray()      # Graph 2
    
    # Store in lists for product()
    distinct_Xs = [X_raw]
    distinct_As = [A_raw, B_raw]

    labels = data["label"]
# Check if it is a Row Vector (1, N) or Column Vector (N, 1)
    if labels.shape[0] == 1 or labels.shape[1] == 1:
        labels = labels.flatten()
    else:
        # Only use argmax if it's actually One-Hot (N, Classes)
        # e.g. shape is (3025, 3)
        labels = np.argmax(labels, axis=1)
    # --- FIX ENDS HERE ---
        
    return distinct_As, distinct_Xs, labels

if __name__ == "__main__":
    # 1. Load Raw Lists
    raw_As, raw_Xs, labels = load_data("tests/datasets/ACM.mat")
    labels = np.asarray(labels).reshape(-1)
    k = len(np.unique(labels))

    # 2. Use 'product' to generate views
    # This creates: [(PAP, X), (PLP, X)]
    view_combinations = list(product(raw_As, raw_Xs))
    
    print(f"Generated {len(view_combinations)} views using product().")

    # 3. Unzip them for the LMGEC .fit() method
    # .fit() needs a list of Xs and a list of Adjs
    final_As = [view[0] for view in view_combinations]
    final_Xs = [view[1] for view in view_combinations]

    # 4. Train
    model = LMGEC(
        n_clusters=k,
        embedding_dim=k+1,
        max_iter=30,
        temperature=10.0,
        tf_idf=True,
        center_data=True
    )

    print("Training...")
    model.fit(final_Xs, adjs=final_As, y_true=labels)

    # 5. Evaluate
    print("\n--- Results ---")
    print(f"Accuracy: {model.accuracy():.4f}")
    print(f"NMI:      {model.nmi():.4f}")
    print(f"F1:       {model.f1_score():.4f}")