from pathlib import Path
import numpy as np

root = Path(__file__).parent.parent.parent

for sub in (root / "results").glob("sub-1*"):
    fname = sub / f"{sub.name}-decoding.npy"
    if fname.exists():
        result = np.load(fname, allow_pickle=True)
