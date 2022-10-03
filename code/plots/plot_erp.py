from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mne import read_evokeds
from mne.viz import plot_topomap


root = Path(__file__).parent.parent.parent.absolute()
plt.style.use(["science", "no-latex"])

topo_width = 1  # number of samples to average for a topo plot
topo_times = [1.1, 1.2, 1.3, 1.4]  # times for the EEG topoplot
ftopo_time = [1.2, 1.5]  # start and stop for the f-score topoplot

evoked = read_evokeds(root / "results" / "grand_averageII-evo.fif")[0]
fz = np.where(np.asarray(evoked.info["ch_names"]) == "Fz")[0][0]
topo_idx = [np.argmin(np.abs(t - evoked.times)) for t in topo_times]
f_idx = [np.argmin(np.abs(t - evoked.times)) for t in ftopo_time]
significance = np.zeros(evoked.data.shape)
# calculate the average F-statistic and significant temporal clusters
subfolders = list((root / "results").glob("sub-1*"))
for subfolder in subfolders:
    results = np.load(
        subfolder / f"{subfolder.name}_cluster.npy", allow_pickle=True
    ).item()
    sig_mask = np.zeros(evoked.data.shape)
    idx = np.where(results["clusters_p"] < 0.05)[0]
    for i in idx:
        cluster = results["clusters"][i]
        for t, ch in zip(cluster[0], cluster[1]):
            sig_mask[ch, t] = 1
    significance += sig_mask


fig, ax = plt.subplot_mosaic(
    [["a1", "a2", "a3", "a4"], ["b1", "b1", "b1", "b1"], ["c1", "c1", "c1", "c1"]],
    figsize=(10, 6),
)
for ichan in range(evoked.data.shape[0]):
    ax["b1"].plot(
        evoked.times, evoked.data[ichan, :] * 1e6, color="gray", linewidth=0.3
    )
ax["b1"].plot(  # highlight one channel
    evoked.times, evoked.data[fz, :] * 1e6, color="black", linewidth=1.5
)

ax["b1"].axvline(x=0, ymin=-3, ymax=3, color="red")
ax["b1"].axvline(x=1, ymin=-3, ymax=3, color="red")

for ti, axname in zip(topo_idx, ["a1", "a2", "a3", "a4"]):
    data = evoked.data[:, ti - topo_width : ti + 1 + topo_width].mean(axis=-1)
    plot_topomap(data, evoked.info, axes=ax[axname], show=False)
    ax["b1"].axvline(x=evoked.times[ti], ymin=-3, ymax=3, color="black", linestyle="--")

xgrid = evoked.times
ygrid = np.arange(evoked.data.shape[0])
im = ax["c1"].pcolormesh(xgrid, ygrid, significance)
# plot the colorbar
cax = fig.add_axes([0.905, 0.11, 0.02, 0.245])
fig.colorbar(im, cax=cax, orientation="vertical")

# axis labeling
ax["c1"].sharex(ax["b1"])
ax["b1"].set(ylabel="Amplitude [\u03BCV]")
plt.setp(ax["b1"].get_xticklabels(), visible=False)
ax["c1"].set(yticks=[], xlabel="Time [s]")
plt.savefig(root / "paper" / "figures" / "erp.png", dpi=800)
