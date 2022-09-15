"""
Plot the elevation gain for the localization tests as well as
for the behavioral task in experiment II.
"""
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

root = Path(__file__).parent.parent.absolute()
plt.style.use("science")


def line(x, a, b):
    return a + b * x


fig, ax = plt.subplots(1, 3)
b_loc1, b_loc2, b_exp2 = [], [], []
a_loc1, a_loc2, a_exp2 = [], [], []
for subfolder in (root / "bids").glob("sub*"):
    if int(subfolder.name[-3:]) < 100:
        fname = subfolder / "beh" / f"{subfolder.name}_task-loctest_beh.tsv"
        if fname.exists():
            data = np.loadtxt(fname)
    else:
        # data = np.loadtxt(subfolder / "beh" / f"{subfolder.name}_task-loctest_beh.tsv")

        data = np.genfromtxt(
            subfolder / "beh" / f"{subfolder.name}_task-oneback_beh.tsv",
            skip_header=1,
            usecols=(1, 2),
        )
        data = data[~np.isnan(data[:, 1])]
        b, a, _, _, _ = linregress(data[:, 0], data[:, 1])
        b_exp2.append(b)
        a_exp2.append(a)
        x = np.unique(data[:, 0])
        ax[2].plot(x, line(x, a, b), color="gray", linewidth=0.5)
