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
        ybar = -1.4
        xticknames = np.arange(-0.6, 0.5, 0.2)
        subjects = list((root / "results").glob("sub-0*"))
        legloc = (0.4, 0.07)
    else:
        adapter_dur = 1.0
        probe_dur = 0.1
        tmin, tmax = 0.15, 0.9  # time interval for ftopo, regression
        ybar = -3.5
        xticknames = np.arange(-1.0, 1, 0.3)
        subjects = list((root / "results").glob("sub-1*"))
        legloc = (0.25, 0.07)

    clusters = np.loadtxt(root / "results" / f"clusters{exp}.csv")
    clusters = clusters[clusters[:, 1] < 0.05]  # select significant clusters
    ftopo = np.zeros(64)
    # compute the average F-value between tmin and tmax:
    for sub in subjects:
        results = np.load(sub / f"{sub.name}_cluster.npy", allow_pickle=True).item()
        nmin = np.argmin(np.abs(adapter_dur + tmin - results["t"]))
        nmax = np.argmin(np.abs(adapter_dur + tmax - results["t"]))
        ftopo += results["f"][nmin:nmax, :].mean(axis=0)
    ftopo /= len(subjects)
    tuning = np.load(root / "results" / f"elevation_tuning_exp{exp}.npy")
    tuning_mean, tuning_std = tuning.mean(axis=0), tuning.std(axis=0)
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
        [["1", "1", "1", "2"], ["1", "1", "1", "3"]], figsize=(8, 4)
    )
    plt.subplots_adjust(wspace=0.25, hspace=0.1)
    divider = make_axes_locatable(ax["1"])
    ax["5"] = divider.append_axes("bottom", size="6%", pad=0)

    # plot the topomap of f-scores
    mask = np.repeat(False, evoked.info["nchan"])
    mask[ch] = True
    im, _ = plot_topomap(
        ftopo,
        evoked.info,
        show=False,
        axes=ax["2"],
        mask=mask,
        vlim=(1, None),
    )
    cax = fig.add_axes([0.91, 0.65, 0.01, 0.2])
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label("F-score")

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
    ax["1"].legend(loc=legloc)
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
    cax = fig.add_axes([0.7, 0.11, 0.01, 0.048])
    fig.colorbar(
        im, cax=cax, orientation="vertical", ticks=[0, int(significance.max())]
    )
    cax.set_title("N", fontsize="medium")
    cax.tick_params(labelsize=8)
    xticks = [np.argmin(np.abs(evoked.times - adapter_dur - t)) for t in xticknames]
    ax["5"].set(
        yticks=[], xlabel="Time [s]", xticks=xticks, xticklabels=xticknames.round(1)
    )
    ax["1"].axvline(x=0, ymin=0, ymax=1, color="black", linestyle="--")
    ax["1"].axvline(x=-adapter_dur, ymin=0, ymax=1, color="gray", linestyle="--")
    if exp == "I":
        x = np.linspace(25, 75, tuning.shape[-1])
        ax["3"].plot(x, tuning_mean.mean(axis=0), label="37.5\u00b0", color="black")
        ax["3"].fill_between(
            x,
            tuning_mean.mean(axis=0) + 2 * tuning_std.mean(axis=0),
            tuning_mean.mean(axis=0) - 2 * tuning_std.mean(axis=0),
            alpha=0.3,
            color="black",
        )
        ax["3"].plot(x, tuning_mean[0], label="-37.5\u00b0", color=colors[6])
        ax["3"].plot(x, tuning_mean[1], label="-37.5\u00b0", color=colors[6])

        ax["3"].set(
            ylabel="Mean abs. amplitude [\u03BCV]",
            xlabel="Adapter-probe distance [\u00b0]",
        )

        ax["3"].annotate(
            "*",
            xy=(30, tuning_mean[0, 5]),
            xycoords="data",
            xytext=(50, -20),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2"),
        )
        ax["3"].annotate(
            "n.s.",
            xy=(50, tuning_mean[1, 49]),
            xycoords="data",
            xytext=(-40, 40),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
        )

    else:
        x = np.linspace(-37.5, 37.5, tuning.shape[-1])
        ax["3"].plot(x, tuning_mean, color="black")
        ax["3"].fill_between(
            x,
            tuning_mean + 2 * tuning_std,
            tuning_mean - 2 * tuning_std,
            alpha=0.3,
            color="black",
        )
        ax["3"].set(
            ylabel="Mean amplitude [\u03BCV]",
            xlabel="Elevation [\u00b0]",
        )
        ax["3"].text(0, 0.2, "***")

    ax["3"].yaxis.tick_right()
    ax["3"].yaxis.set_label_position("right")

    # subplot annotations
    for key, axis, x in zip(
        ["A", "B", "C"], [ax["1"], ax["2"], ax["3"]], [-0.04, -0.1, -0.1]
    ):
        axis.text(x, 1, key, transform=axis.transAxes, size=10, weight="bold")

    fig.savefig(
        root / "results" / "plots" / f"erp_exp{exp}.png", dpi=300, bbox_inches="tight"
    )
