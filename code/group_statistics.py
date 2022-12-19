from pathlib import Path
import numpy as np
from mne import read_epochs

root = Path(__file__).parent.parent.absolute()


def get_ftopo(tmin, tmax, exp="I"):
    if exp == "I":
        subfolders = list((root / "results").glob("sub-0*"))
    elif exp == "II":
        subfolders = list((root / "results").glob("sub-1*"))
    else:
        raise ValueError('exp must be "I" or "II"')
    topo = np.zeros(64)
    for sub in subfolders:
        results = np.load(sub / f"{sub.name}_cluster.npy", allow_pickle=True).item()
        f, t = results["f"], results["t"]  # f-values and time points
        nmin, nmax = np.argmin(np.abs(tmin - t)), np.argmin(np.abs(tmax - t))
        topo += f[nmin:nmax, :].mean(axis=0)
    return topo / len(subfolders)


def get_encoding(ch, tmin, tmax, exp="I"):
    if exp == "I":
        subfolders = list((root / "preprocessed").glob("sub-0*"))
        est = lambda x: np.mean(np.abs(x))
    elif exp == "II":
        subfolders = list((root / "preprocessed").glob("sub-1*"))
        est = lambda x: np.mean(x)
    else:
        raise ValueError('exp must be "I" or "II"')

    probe, adapter, amplitude = [], [], []
    for isub, sub in enumerate(subfolders):
        epochs = read_epochs(sub / f"{sub.name}-epo.fif")
        keys = [key for key in epochs.event_id.keys() if not key == "deviant"]
        for key in keys:
            evoked = epochs[key].average()
            nmin = np.argmin(np.abs(tmin - evoked.times))
            nmax = np.argmin(np.abs(tmax - evoked.times))
            if isinstance(ch, str):
                evoked.pick_channels([ch])
                data = evoked.data.flatten() * 1e6
            else:
                data = evoked.data[ch, :] * 1e6
            amp = est(data[nmin:nmax])
            if exp == "I":
                ad, pr = float(key.split()[0]), float(key.split()[2])
            else:
                ad, pr = float(key), 0
            amplitude.append(amp), probe.append(pr), adapter.append(ad)
    return np.stack([adapter, probe, amplitude])
