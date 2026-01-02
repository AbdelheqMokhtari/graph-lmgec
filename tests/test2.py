import scipy.io as io
from pathlib import Path
import numpy as np

def inspect_mat_file(file_path):
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File '{path}' not found.")
        return

    print(f"--- Inspecting: {path} ---")
    try:
        data = io.loadmat(str(path))
    except Exception as e:
        print(f"Failed to load .mat file: {e}")
        return

    # Loop through all keys
    for key, value in data.items():
        # Skip internal MATLAB metadata (starts with __)
        if key.startswith("__"):
            continue
            
        # Get info about the item
        obj_type = type(value)
        
        # specific handling to show shapes of arrays/matrices
        if isinstance(value, np.ndarray):
            print(f"Key: '{key}' | Shape: {value.shape} | Type: {value.dtype}")
        elif hasattr(value, 'shape'):
            # For sparse matrices (scipy.sparse)
            print(f"Key: '{key}' | Shape: {value.shape} | Type: Sparse Matrix")
        else:
            print(f"Key: '{key}' | Type: {obj_type}")

if __name__ == "__main__":
    # Change this to your actual file path
    inspect_mat_file("tests/datasets/ACM.mat")