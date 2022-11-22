from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter
from scipy.stats import linregress
from mne import read_evokeds


def convna(x):
    if x == b"n/a":
        return np.nan
    else:
        return float(x)
    return


plt.style.use(["science", "no-latex"])
root = Path(__file__).parent.parent.parent
subjects = list((root / "results").glob("sub-1*"))
n_permute = 1000
evoked = read_evokeds(root / "results" / "grand_averageII-evo.fif")[0]
# get time window for calculating average decoding accuracy
idx_acc = (np.argmin(np.abs(evoked.times - 1.2)), np.argmin(np.abs(evoked.times - 1.5)))

group_data = {  # actual data
    "37.5 vs 12.5": np.zeros((27, 205)),
    "37.5 vs -12.5": np.zeros((27, 205)),
    "37.5 vs -37.5": np.zeros((27, 205)),
    "12.5 vs -12.5": np.zeros((27, 205)),
    "12.5 vs -37.5": np.zeros((27, 205)),
    "-12.5 vs -37.5": np.zeros((27, 205)),
}
group_data_resampled = {  # data after bootstrapping
    "37.5 vs 12.5": np.zeros((n_permute, 205)),
    "37.5 vs -12.5": np.zeros((n_permute, 205)),
    "37.5 vs -37.5": np.zeros((n_permute, 205)),
    "12.5 vs -12.5": np.zeros((n_permute, 205)),
    "12.5 vs -37.5": np.zeros((n_permute, 205)),
    "-12.5 vs -37.5": np.zeros((n_permute, 205)),
}
# get each subjects decoding result and average across splits
count = 0
for i, sub in enumerate(subjects):
    fname = sub / f"{sub.name}-decoding.npy"
    if fname.exists():
        result = np.load(fname, allow_pickle=True).item()
        for key in result:
            group_data[key][count, :] = result[key].mean(axis=0)
        count += 1
for key in group_data:  # resample
    for ip in range(n_permute):
        idx = np.random.choice(count, count, replace=True)
        group_data_resampled[key][ip, :] = group_data[key][idx, :].mean(axis=0)

# get average decoding accuracy per subject
acc = []
for i in range(count):
    sub_acc = []
    for key in group_data:
        sub_acc.append(group_data[key][i, idx_acc[0] : idx_acc[1]].mean())
    acc.append(np.mean(sub_acc))


eg_test, eg_task = [], []  # elevation gain
for subfolder in (root / "bids").glob("sub-1*"):
    fname = subfolder / "beh" / f"{subfolder.name}_task-loctest_beh.tsv"
    if fname.exists():
        data = np.loadtxt(fname, skiprows=1, converters={0: convna, 1: convna})
        data = data[~np.isnan(data[:, 1])]
        b, _, _, _, _ = linregress(data[:, 0], data[:, 1])
        eg_test.append(b)
    fname = subfolder / "beh" / f"{subfolder.name}_task-oneback_beh.tsv"
    if fname.exists():
        data = np.genfromtxt(fname, skip_header=1, usecols=(1, 2))
        data = data[~np.isnan(data[:, 1])]
        b, _, _, _, _ = linregress(data[:, 0], data[:, 1])
        eg_task.append(b)

fig, ax = plt.subplot_mosaic([["a1", "a1"], ["b1", "b2"], ["c1", "c1"]])
divider = make_axes_locatable(ax["a1"])
ax["a2"] = divider.append_axes("right", size="100%", pad=0.1)
for key, data in group_data_resampled.items():
    mean, std = data.mean(axis=0), data.std(axis=0)
    # apply filter for smoothing
    mean, std = savgol_filter(mean, 10, 5), savgol_filter(std, 20, 5)
    for axkey in ["a1", "a2"]:
        ax[axkey].plot(evoked.times, mean, label=key)
        ax[axkey].fill_between(evoked.times, mean + std, mean - std, alpha=0.3)
ax["a1"].legend(loc="upper left", fontsize="x-small", ncol=2)
ax["a1"].spines["right"].set_visible(False)
ax["a2"].spines["left"].set_visible(False)
ax["a1"].set(
    ylabel="Accuracy [a.u.c]",
    xlabel="Time [s]",
    xlim=(-0.1, 0.4),
    xticks=[-0.1, 0, 0.1, 0.2, 0.3],
)
ax["a1"].xaxis.set_label_coords(1.0, -0.1)
ax["a2"].set(yticks=[], xlim=(1.0, 1.5), xticks=[1.1, 1.2, 1.3, 1.4, 1.5])
ax["a1"].yaxis.tick_left()
# diagonal lines to visualize axis interruption
d = 0.015  # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=ax["a1"].transAxes, color="k", clip_on=False)
ax["a1"].plot((1 - d / 2, 1 + d / 2), (-d, +d), **kwargs)
ax["a1"].plot((1 - d / 2, 1 + d / 2), (1 - d, 1 + d), **kwargs)

kwargs.update(transform=ax["a2"].transAxes)  # switch to the bottom axes
ax["a2"].plot((-d / 2, +d / 2), (1 - d, 1 + d), **kwargs)
ax["a2"].plot((-d / 2, +d / 2), (-d, +d), **kwargs)

ax["b1"].hist(eg_test, label="test", alpha=0.7)
ax["b1"].hist(eg_task, label="task", alpha=0.7)
ax["b1"].legend(loc="upper left", fontsize="small")

ax["b2"].hist(acc)
