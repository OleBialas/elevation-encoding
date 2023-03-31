from pathlib import Path
import numpy as np
from scipy.stats import linregress
from mne import read_epochs


def line(a, b, x):
    return a + b * x


root = Path(__file__).parent.parent.absolute()
n_resample = 10000  # number of resampling for tuning curve
n_elevations = 100  # points for sampling the linear regression

stats = np.zeros((4, 5), dtype="<U6")
stats[0] = ["test", "slope", "r", "p", "df"]
for exp in ["I", "II"]:
    if exp == "I":
        subs = list((root / "preprocessed").glob("sub-0*"))
        tmin, tmax = 0.7, 0.9
        ch = "Cz"
        x = np.stack([[25, 50, 75, 75, 50, 25] for i in range(len(subs))])
        tuning = np.zeros((n_resample, 3, n_elevations))
    else:
        subs = list((root / "preprocessed").glob("sub-1*"))
        tmin, tmax = 1.15, 1.85
        ch = "FT10"
        x = np.stack([[37.5, 12.5, -12.5, -37.5] for i in range(len(subs))])
        tuning = np.zeros((n_resample, n_elevations))

    for isub, sub in enumerate(subs):
        epochs = read_epochs(sub / f"{sub.name}-epo.fif")
        keys = [key for key in epochs.event_id.keys() if not key == "deviant"]
        if isub == 0:
            data = np.zeros((len(subs), len(keys)))
        epochs.pick_channels([ch])
        evokeds = epochs[keys].average(by_event_type=True)
        del epochs
        for ievo, evo in enumerate(evokeds):
            nmin = np.argmin(np.abs(tmin - evo.times))
            nmax = np.argmin(np.abs(tmax - evo.times))
            if exp == "I":
                data[isub, ievo] = np.mean(np.abs(evo.data.flatten()[nmin:nmax]))
            else:
                data[isub, ievo] = np.mean(evo.data.flatten()[nmin:nmax])
    data *= 1e6

    if exp == "I":
        b, a, r, p, _ = linregress(x[:, :3].flatten(), data[:, :3].flatten())
        stats[1] = [
            "expIau",
            str(b.round(3)),
            str(r.round(3)),
            str(p),
            str(len(data) - 2),
        ]
        b, a, r, p, _ = linregress(x[:, 3:].flatten(), data[:, 3:].flatten())
        stats[2] = [
            "expIad",
            str(b.round(3)),
            str(r.round(3)),
            str(p),
            str(len(data) - 2),
        ]
    else:
        b, a, r, p, _ = linregress(x.flatten(), data.flatten())
        if p < 0.001:
            p = "<0.001"
        else:
            p = str(p.round(3))
        stats[3] = ["expII", str(b.round(3)), str(r.round(3)), p, str(len(data) - 2)]

    # resample and compute tuning curves
    for i in range(n_resample):
        idx = np.random.choice(len(data), len(data))
        resampled = data[idx, :]
        if exp == "I":
            b, a, r, p, _ = linregress(x.flatten(), resampled.flatten())
            tuning[i, 0, :] = line(a, b, np.linspace(x.min(), x.max(), n_elevations))
            b, a, r, p, _ = linregress(x[:, :3].flatten(), resampled[:, :3].flatten())
            tuning[i, 1, :] = line(a, b, np.linspace(x.min(), x.max(), n_elevations))
            b, a, r, p, _ = linregress(x[:, 3:].flatten(), resampled[:, 3:].flatten())
            tuning[i, 2, :] = line(a, b, np.linspace(x.min(), x.max(), n_elevations))
        else:
            b, a, r, p, _ = linregress(x.flatten(), resampled.flatten())
            tuning[i, :] = line(a, b, np.linspace(x.min(), x.max(), n_elevations))
    np.save(root / "results" / f"elevation_tuning_exp{exp}.npy", tuning)
np.savetxt(root / "results" / "elevation_tuning_stats.csv", stats, fmt="%s")
