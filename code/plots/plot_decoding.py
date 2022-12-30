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


def line(x, a, b):
    return a + x * b


root = Path(__file__).parent.parent.parent
plt.style.use(["science", "no-latex"])
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig, ax = plt.subplot_mosaic([["A", "B"]], figsize=(8, 4))
divider = make_axes_locatable(ax["B"])
ax["C"] = divider.append_axes("top", size="20%", pad=0)
ax["D"] = divider.append_axes("right", size="20%", pad=0)

evoked = read_evokeds(root / "results" / "grand_averageII-ave.fif")[0]
n_permute = 10000
tmin, tmax = 0.15, 0.9
adapter_dur = 1.0
plt.style.use(["science", "no-latex"])
subjects = list((root / "results").glob("sub-1*"))
# calculate average decoding accros cross validation folds per subject
for isub, sub in enumerate(subjects):
    results = np.load(sub / f"{sub.name}-decoding.npy", allow_pickle=True).item()
    if isub == 0:
        keys = [key for key in results.keys()]
        decoding_data = np.zeros((len(subjects), len(keys), results[keys[0]].shape[-1]))
    for ikey, key in enumerate(keys):
        decoding_data[isub, ikey, :] = results[key].mean(axis=0)

# get each subjects elevation gain
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
eg_task = np.asarray(eg_task)
eg_task[eg_task < 0] = 0  # set negative EGs to 0

# get each subjects mean decoding accuracy between tmin and tmax
nmin = np.argmin(np.abs(adapter_dur + tmin - evoked.times))
nmax = np.argmin(np.abs(adapter_dur + tmax - evoked.times))
avg_acc = decoding_data[:, :, nmin:nmax].mean(axis=(1, 2))

# calculate regression between eg and acc with bootstrapping
x = np.linspace(0, 1, 100)
resampled_task = np.zeros((n_permute, len(x)))
resampled_test = np.zeros((n_permute, len(x)))
for ip in range(n_permute):
    idx = np.random.choice(len(avg_acc), len(avg_acc))
    b, a, r, p, _ = linregress(np.asarray(eg_task)[idx], np.asarray(avg_acc)[idx])
    resampled_task[ip] = line(x, a, b)
    b, a, r, p, _ = linregress(np.asarray(eg_test)[idx], np.asarray(avg_acc)[idx])
    resampled_test[ip] = line(x, a, b)

# plot the data
for ikey, key in enumerate(keys):
    ax["A"].plot(
        evoked.times - adapter_dur,
        savgol_filter(decoding_data.mean(axis=0)[ikey], 13, 5),
        label=key,
    )
ax["A"].legend(loc="upper left")
ax["A"].set(
    xlabel="Time [s]", ylabel="Accuracy [a.u.c.]", yticks=[0.5, 0.55, 0.6, 0.65]
)

for i, data in enumerate([resampled_test, resampled_task]):
    if i == 0:
        eg = eg_test
    else:
        eg = eg_task
    mean, std = data.mean(axis=0), data.std(axis=0)
    ax["B"].scatter(eg, avg_acc, color=colors[i])
    ax["B"].plot(x, mean, color=colors[i])
    ax["B"].fill_between(x, mean + 2 * std, mean - 2 * std, alpha=0.2, color=colors[i])
    ax["B"].set(
        xlabel="Elevation gain [a.u.]",
        ylabel="Mean accuracy [a.u.c.]",
        xlim=(0, 1),
        ylim=(0.45, 0.75),
        yticks=[0.5, 0.55, 0.6, 0.65, 0.7],
    )

ax["B"].text(0.15, 0.56, "**")
ax["B"].text(0.22, 0.48, "n.s.")
ax["C"].hist(eg_test, alpha=0.5, bins=20, color=colors[0])
ax["C"].hist(eg_task, alpha=0.5, bins=20, color=colors[1])
ax["C"].set(xlim=(0, 1), xticks=[], yticks=[2])
ax["D"].hist(avg_acc, bins=20, orientation="horizontal", color="gray", alpha=0.5)
ax["D"].set(ylim=(0.45, 0.75), yticks=[], xticks=[2])

plt.tight_layout()
fig.text(0.95, 0.85, "number\n of\n subjects", ha="center")

fig.text(0.08, 0.87, "A", size=10, weight="bold")
fig.text(0.51, 0.87, "B", size=10, weight="bold")
fig.savefig(root / "results" / "plots" / "decoding.png", dpi=300, bbox_inches="tight")
