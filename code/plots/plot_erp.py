""" For each experiment, plot the average evoked response a color bar indicating
the number of significant clusters across time, the topo map of F-scores and the
ERP for each condition at the most significant channel.
"""
import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt, patches
import matplotlib.transforms as mtransforms
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import linregress
from mne import read_evokeds
from mne.viz import plot_topomap

root = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(root / "code"))
from group_statistics import get_ftopo, get_encoding

plt.style.use(["science", "no-latex"])
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
ramp_dur = 1  # duration of ramp for adapter and probe in image


def line(a, b, x):
    return a + b * x


for exp in ["I", "II"]:
    evoked = read_evokeds(root / "results" / f"grand_average{exp}-ave.fif")[0]
    conditions = read_evokeds(
        root / "results" / f"grand_average{exp}_conditions-ave.fif"
    )
    if exp == "I":
        evoked.crop(None, 1.0)
        [con.crop(None, 1.0) for con in conditions]
        adapter_dur = 0.6
        probe_dur = 0.15
        tmin, tmax = 0.1, 0.3  # time interval for ftopo, regression
    else:
        adapter_dur = 1.0
        probe_dur = 0.1
        tmin, tmax = 0.15, 0.9  # time interval for ftopo, regression
        ybar = -3.5

    clusters = np.loadtxt(root / "results" / f"clusters{exp}.csv")
    clusters = clusters[clusters[:, 1] < 0.05]  # select significant clusters
    # ftopo = get_ftopo(tmin, tmax, exp)
    ftopo = np.load(root / "results" / f"ftopo{exp}.npy")

    ch = np.argmax(ftopo)  # channel with the largest effect
    [
        con.savgol_filter(20).pick_channels([evoked.info["ch_names"][ch]])
        for con in conditions
    ]
    # color bar indicating significant clusters over time
    significance = np.zeros((1, len(evoked.times)))
    for i, t in enumerate(evoked.times):
        significance[0, i] = np.logical_and(
            t >= clusters[:, 2], t <= clusters[:, 3]
        ).sum()

    fig, ax = plt.subplot_mosaic(
        [["1", "1", "1", "2"], ["1", "1", "1", "3"]], figsize=(10, 6)
    )
    divider = make_axes_locatable(ax["1"])
    ax["5"] = divider.append_axes("bottom", size="8%", pad=0)

    # plot the topomap of f-scores
    mask = np.repeat(False, evoked.info["nchan"])
    mask[ch] = True
    plot_topomap(
        ftopo,
        evoked.info,
        show=False,
        axes=ax["2"],
        mask=mask,
    )

    for ichan in range(evoked.data.shape[0]):
        ax["1"].plot(
            evoked.times - adapter_dur,
            evoked.data[ichan, :] * 1e6,
            color="gray",
            linewidth=0.3,
        )
    if exp == "I":
        for con in conditions:
            adapter, probe = float(con.comment.split()[0]), float(
                con.comment.split()[2]
            )
            if adapter == 37.5:
                ax["1"].plot(
                    evoked.times - adapter_dur,
                    con.data.flatten() * 1e6,
                    linewidth=2,
                    label=f"{probe}\u00b0",
                )
    else:
        for con in conditions:
            probe = con.comment
            ax["1"].plot(
                evoked.times - adapter_dur,
                con.data.flatten() * 1e6,
                linewidth=2,
                label=f"{probe}\u00b0",
            )
    ax["1"].legend(loc="upper right")
    ax["1"].hlines(y=ybar, xmin=tmin, xmax=tmax, color="black", linewidth=2)
    ax["1"].set(
        ylabel="Amplitude [\u03BCV]",
        xlim=((evoked.times - adapter_dur).min(), (evoked.times - adapter_dur).max()),
    )
    ax["1"].tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        labelbottom=False,
    )  # labels along the bottom edge are off
    im = ax["5"].imshow(significance, aspect="auto")
    cax = fig.add_axes([0.7, 0.11, 0.01, 0.055])
    fig.colorbar(im, cax=cax, orientation="vertical")
    xticknames = np.arange(-0.7, 0.4, 0.2)
    xticks = [np.argmin(np.abs(evoked.times - adapter_dur - t)) for t in xticknames]
    ax["5"].set(
        yticks=[], xlabel="Time [s]", xticks=xticks, xticklabels=xticknames.round(1)
    )
    ax["1"].axvline(x=0, ymin=0, ymax=1, color="black", linestyle="--")
    ax["1"].axvline(x=-adapter_dur, ymin=0, ymax=1, color="gray", linestyle="--")
