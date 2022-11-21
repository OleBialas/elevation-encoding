from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt, patches
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
xticknames = np.arange(0, 1.6, 0.2)
xticks = [np.argmin(np.abs(evoked.times - t)) for t in xticknames]
fz = np.where(np.asarray(evoked.info["ch_names"]) == "Cz")[0][0]
mask = np.repeat(False, 64)
mask[fz] = True
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
significance = significance.max(axis=0, keepdims=True)  # collapse across channels


fig, ax = plt.subplot_mosaic(
    [["a1", "a2", "a3", "a4"], ["b1", "b1", "b1", "b1"], ["b1", "b1", "b1", "b1"]],
    figsize=(10, 6),
)
divider = make_axes_locatable(ax["b1"])
ax["c1"] = divider.append_axes("bottom", size="15%", pad=0)

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
    plot_topomap(
        data,
        evoked.info,
        axes=ax[axname],
        show=False,
        mask_params=dict(marker="x", markersize=14, linewidth=3),
        mask=mask,
    )
    ax["b1"].axvline(x=evoked.times[ti], ymin=-3, ymax=3, color="black", linestyle="--")

# xgrid = evoked.times
# ygrid = np.arange(evoked.data.shape[0])
# im = ax["c1"].pcolormesh(xgrid, ygrid, significance)
im = ax["c1"].imshow(significance, aspect="auto")
# plot the colorbar
cax = fig.add_axes([0.905, 0.11, 0.01, 0.07])
fig.colorbar(im, cax=cax, orientation="vertical")

# axis labeling
ax["b1"].set(
    ylabel="Amplitude [\u03BCV]", xlim=(evoked.times.min(), evoked.times.max())
)
ax["c1"].set(
    yticks=[], xlabel="Time [s]", xticks=xticks, xticklabels=xticknames.round(1)
)

# figure labeling
for label, axes in zip(["A", "B", "C", "D", "E", "F"], list(ax.values())):
    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    axes.text(
        0.0,
        1.0,
        label,
        transform=axes.transAxes + trans,
        fontsize="medium",
        verticalalignment="top",
        fontfamily="serif",
    )

# Draw lines connecting the topo-plots with ERP trace
for i, t in enumerate(topo_times):
    arrow = patches.ConnectionPatch(
        [0, -0.1],
        [t, 4.2],
        coordsA=ax[f"a{i+1}"].transData,
        coordsB=ax["b1"].transData,
        # Default shrink parameter is 0 so can be omitted
        color="black",
    )
    fig.patches.append(arrow)


plt.savefig(root / "paper" / "figures" / "erp.png", dpi=800)
