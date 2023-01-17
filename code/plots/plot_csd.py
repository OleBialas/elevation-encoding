from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt, patches
import matplotlib.transforms as mtransforms
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mne import read_evokeds
from mne.viz import plot_topomap

root = Path(__file__).parent.parent.parent.absolute()
plt.style.use(["science", "no-latex"])
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

csd = read_evokeds(root / "results" / "group_csd-ave.fif")

[c.resample(128).crop(0.9, 2.0) for c in csd]

times = csd[0].times - 1.0
topo_times = [1.2, 1.3, 1.4, 1.5, 1.6]
chs = ["T7", "T8", "Fz"]
linestyles = ["solid", "dotted", "dashed"]
ch_idx = [np.where(np.array(csd[0].info["ch_names"]) == ch)[0][0] for ch in chs]
vmin, vmax = -1.9, 1.6
fig, ax = plt.subplot_mosaic(
    [
        ["1a", "2a", "3a", "4a", "5a"],
        ["1b", "2b", "3b", "4b", "5b"],
        ["1c", "2c", "3c", "4c", "5c"],
        ["1d", "2d", "3d", "4d", "5d"],
        ["e", "e", "e", "e", "e"],
        ["e", "e", "e", "e", "e"],
    ]
)

ax["e"].plot(times, csd[-1].data.T * 1e3, color="black", linewidth=0.3)
for linestyle, idx in zip(linestyles, ch_idx):
    ax["e"].plot(
        times,
        csd[0].data[idx, :] * 1e3,
        color=colors[0],
        linestyle=linestyle,
        linewidth=2,
    )
    ax["e"].plot(
        times,
        csd[3].data[idx, :] * 1e3,
        color=colors[3],
        linestyle=linestyle,
        linewidth=2,
    )
ax["e"].set(xlim=(times.min(), times.max()))

lables = ["a", "b", "c", "d"]
for it, t in enumerate(topo_times):
    for i in range(4):
        axes = f"{it+1}{lables[i]}"
        csd[i].plot_topomap(
            t, axes=ax[axes], show=False, colorbar=False, vlim=(vmin, vmax)
        )
        if i > 0:
            ax[axes].set(title=None)

plt.subplots_adjust(wspace=-0.4, hspace=-0.05)
fig.sa
