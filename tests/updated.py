import numpy as np
from graphs.lmgec import LMGEC  # Importing from your new package!

# 1. Generate Fake Data for Testing
# View 1: 100 samples, 50 features
X1 = np.random.rand(100, 50)
# View 2: 100 samples, 20 features
X2 = np.random.rand(100, 20)
Xs = [X1, X2]

# Fake Labels (3 clusters)
labels = np.random.randint(0, 3, 100)

# 2. Initialize Model
print("Initializing LMGEC...")
model = LMGEC(n_clusters=3, embedding_dim=10, max_iter=5)

# 3. Fit (with adjacency as None for simplicity here)
print("Fitting...")
model.fit(Xs, adjs=[None, None], y_true=labels)

# 4. Use your new accessor methods
print(f"Accuracy: {model.accuracy():.4f}")
print(f"NMI:      {model.nmi():.4f}")
print(f"Silhouette: {model.silhouette():.4f}")

print("\nSuccess! The package is working.")



