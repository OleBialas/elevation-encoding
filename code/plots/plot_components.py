from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms
from scipy.signal import savgol_filter
from mne import read_evokeds
from mne.viz import plot_topomap

root = Path(__file__).parent.parent.parent.absolute()

fig = plt.figure()
gs = gridspec.GridSpec(4, 8)
ax = {}
ax["A"] = plt.subplot(gs[1:3, :2])
ax["B"] = plt.subplot(gs[:2, 2:4])
ax["B2"] = plt.subplot(gs[2:, 2:4])
ax["C"] = plt.subplot(gs[:2, 4:6])
ax["C2"] = plt.subplot(gs[2:, 4:6])
ax["D"] = plt.subplot(gs[:2, 6:])
ax["D2"] = plt.subplot(gs[2:, 6:])

csd = read_evokeds(root / "results" / "group_csd-ave.fif")
X = csd[-1].data
X -= X.mean(axis=1, keepdims=True)  # subtract each channels mean
cov = (X @ X.T) / (X.shape[1] - 1)

eig_vals, eig_vecs = np.linalg.eig(cov)
# scale eigenvalues to percent variance explained
eig_vals = (eig_vals / eig_vals.sum()) * 100
ax["A"].plot(eig_vals, color="black", linewidth=1.5)
ax["A"].set(ylabel="Variance explained [%]", xlabel="Component", xlim=(0, 20))

labels = ["B", "C", "D"]
for i in range(3):
    plot_topomap(eig_vecs[:, i], csd[0].info, show=False, axes=ax[labels[i]])
    for j in range(4):
        X = csd[j].data
        stc = X.T @ eig_vecs[:, i]
        stc = savgol_filter(stc, 50, 8)
        ax[labels[i] + "2"].plot(csd[0].times - 1.0, stc, label=csd[j].comment)
        ax[labels[i] + "2"].set(yticks=[])
        ax[labels[i] + "2"].set_title(
            f"PC{i+1}: {eig_vals[i].round(1)}%", fontsize="medium"
        )
ax["C2"].set_xlabel("Time [s]")
ax["B2"].legend(loc="lower right", fontsize="xx-small")
ax["D2"].set_ylabel("Amplitude [a.u.]")
ax["D2"].yaxis.set_label_position("right")


for label in ["A", "B", "C", "D"]:
    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    ax[label].text(
        0.0,
        1.0,
        label,
        transform=ax[label].transAxes + trans,
        fontsize=10,
        verticalalignment="top",
        fontfamily="serif",
    )

plt.savefig(root / "results" / "plots" / "components.png", dpi=300, bbox_inches="tight")
