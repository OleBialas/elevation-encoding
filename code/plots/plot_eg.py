"""
Plot the elevation gain for the localization tests as well as
for the behavioral task in experiment II.
"""
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms
from scipy.stats import linregress, ttest_rel

root = Path(__file__).parent.parent.parent.absolute()
plt.style.use("science")


def line(x, a, b):
    return a + b * x


def convna(x):
    if x == b"n/a":
        return np.nan
    else:
        return float(x)
    return


fig, ax = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(10, 6))
x = np.array([-50.0, -25.0, 0.0, 25.0, 50.0])  # plot all lines over this interval
b_loc1, b_loc2, b_exp2 = [], [], []
a_loc1, a_loc2, a_exp2 = [], [], []
for subfolder in (root / "bids").glob("sub*"):
    fname = subfolder / "beh" / f"{subfolder.name}_task-loctest_beh.tsv"
    if fname.exists():
        data = np.loadtxt(fname, skiprows=1, converters={0: convna, 1: convna})
        data = data[~np.isnan(data[:, 1])]
        b, a, _, _, _ = linregress(data[:, 0], data[:, 1])
        if int(subfolder.name[-3:]) < 100:
            a_loc1.append(a)
            b_loc1.append(b)
            ax[0].plot(x, line(x, a, b), color="gray", linewidth=0.5)
        else:
            a_loc2.append(a)
            b_loc2.append(b)
            ax[1].plot(x, line(x, a, b), color="gray", linewidth=0.5)

    fname = subfolder / "beh" / f"{subfolder.name}_task-oneback_beh.tsv"
    if fname.exists():
        data = np.genfromtxt(fname, skip_header=1, usecols=(1, 2))
        data = data[~np.isnan(data[:, 1])]
        b, a, _, _, _ = linregress(data[:, 0], data[:, 1])
        b_exp2.append(b)
        a_exp2.append(a)
        ax[2].plot(x, line(x, a, b), color="gray", linewidth=0.5)

ax[0].plot(x, line(x, np.mean(a_loc1), np.mean(b_loc1)), color="red")
ax[1].plot(x, line(x, np.mean(a_loc2), np.mean(b_loc2)), color="red")
ax[2].plot(x, line(x, np.mean(a_exp2), np.mean(b_exp2)), color="red")

ax[0].set(ylabel="Target Elevation [\u00b0]")
ax[1].set(xlabel="Response Elevation [\u00b0]")


for label, axes in zip(["A", "B", "C"], ax):
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

mean_eg = [np.mean(b_loc1), np.mean(b_loc2), np.mean(b_exp2)]

for eg, axes in zip(mean_eg, ax):
    label = r"$\overline{EG}$" + f"={eg.round(2)}"
    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    axes.text(
        0.6,
        0.1,
        label,
        transform=axes.transAxes + trans,
        fontsize="medium",
        verticalalignment="top",
        fontfamily="serif",
    )

[a.set(adjustable="box", aspect="equal") for a in ax]

print(f"loctest I: EG={np.mean(b_loc1)}, SD={np.std(b_loc1)}")
print(f"loctest II: EG={np.mean(b_loc2)}, SD={np.std(b_loc2)}")
print(f"experiment II: EG={np.mean(b_exp2)}, SD={np.std(b_exp2)}")

print(f"paired samples ttest with {len(b_loc2)} df:")
print(ttest_rel(b_loc2, b_exp2))

plt.savefig(root / "paper" / "figures" / "eg.png", dpi=800)
