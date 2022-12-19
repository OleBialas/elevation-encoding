from pathlib import Path
import numpy as np

root = Path(__file__).parent.parent.absolute()


def ftopo(tmin, tmax, exp="I", standardize=True, fs=128):

    if exp == "I":
        subfolders = (root / "results").glob("sub-0*")
    elif exp == "II":
        subfolders = (root / "results").glob("sub-1*")
    else:
        raise ValueError('exp must be "I" or "II"')
    for sub in subfolders:
        f = np.load(sub / f"{sub.name}_cluster.npy", allow_pickle=True).item()[
            "statistic"
        ]
    times = np.arange(
