from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter
from scipy.stats import linregress, binom
from mne import read_epochs

root = Path(__file__).parent.parent.parent
plt.style.use(["science", "no-latex"])
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def convna(x):
    if x == b"n/a":
        return np.nan
    else:
        return float(x)
    return


def line(x, a, b):
    return a + x * b


fig, ax = plt.subplot_mosaic([["A", "B"]], figsize=(8, 4))
divider = make_axes_locatable(ax["B"])
ax["C"] = divider.append_axes("top", size="20%", pad=0)
ax["D"] = divider.append_axes("right", size="20%", pad=0)

times = read_epochs(root / "preprocessed" / "sub-100" / "sub-100-epo.fif").times
n_permute = 10000
tmin, tmax = 0.15, 0.9
adapter_dur = 1.0
n_obs = 360
thresh = binom.ppf(0.95, n_obs, 0.5) / n_obs  # threshold for p < .05 for one-sided test

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
nmin = np.argmin(np.abs(adapter_dur + tmin - times))
nmax = np.argmin(np.abs(adapter_dur + tmax - times))
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
        times - adapter_dur,
        savgol_filter(decoding_data.mean(axis=0)[ikey], 50, 8),
        label=key,
    )
ax["A"].legend(loc="upper left")
ax["A"].set(
    xlabel="Time [s]",
    ylabel="Accuracy [a.u.c.]",
    yticks=[0.5, 0.55, 0.6, 0.65],
    xlim=(-1, 1.2),
)
ax["A"].axhline(y=thresh, xmin=0, xmax=1, color="black", linestyle="--")

labels = ["Test", "Task"]
for i, data in enumerate([resampled_test, resampled_task]):
    if i == 0:
        eg = eg_test
    else:
        eg = eg_task
    mean, std = data.mean(axis=0), data.std(axis=0)
    ax["B"].scatter(eg, avg_acc, color=colors[i], label=labels[i])
    ax["B"].plot(x, mean, color=colors[i])
    ax["B"].fill_between(x, mean + 2 * std, mean - 2 * std, alpha=0.2, color=colors[i])
    ax["B"].set(
        xlabel="Elevation gain [a.u.]",
        ylabel="Mean accuracy [a.u.c.]",
        xlim=(0, 1),
        ylim=(0.45, 0.75),
        yticks=[0.5, 0.55, 0.6, 0.65, 0.7],
    )

ax["B"].legend()
ax["C"].hist(eg_test, alpha=0.5, bins=20, color=colors[0])
ax["C"].hist(eg_task, alpha=0.5, bins=20, color=colors[1])
ax["C"].set(xlim=(0, 1), xticks=[], yticks=[2])
ax["D"].hist(avg_acc, bins=20, orientation="horizontal", color="gray", alpha=0.5)
ax["D"].set(ylim=(0.45, 0.75), yticks=[], xticks=[2])

plt.tight_layout()
fig.text(0.95, 0.85, "number\n of\n subjects", ha="center")


ax["B"].annotate(
    "**",
    xy=(0.65, resampled_task.mean(axis=0)[65]),
    xycoords="data",
    xytext=(-50, 20),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2"),
)

ax["B"].annotate(
    "n.s.",
    xy=(0.35, resampled_test.mean(axis=0)[35]),
    xycoords="data",
    xytext=(40, -40),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2"),
)
fig.text(0.05, 0.98, "A", fontsize=10, weight="bold")
fig.text(0.55, 0.98, "B", fontsize=10, weight="bold")

fig.savefig(root / "results" / "plots" / "decoding.png", dpi=300, bbox_inches="tight")
