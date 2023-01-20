from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from mne import read_evokeds
from mne.viz import plot_topomap

root = Path(__file__).parent.parent.absolute()

csd = read_evokeds(root / "results" / "group_csd-ave.fif")
X = csd[-1].data
X -= X.mean(axis=1, keepdims=True)  # subtract each channels mean
cov = (X @ X.T) / (X.shape[1] - 1)

eig_vals, eig_vecs = np.linalg.eig(cov)
# scale eigenvalues to percent variance explained
eig_vals = (eig_vals / eig_vals.sum()) * 100


fig, ax = plt.subplots(2, 3)
for i in range(3):
    plot_topomap(eig_vecs[:, i], csd[0].info, show=False, axes=ax[0, i])
    ax[0, i].set(title=f"PC{i+1}: {eig_vals[i].round(1)}%")
    for j in range(4):
        X = csd[j].data
        stc = X.T @ eig_vecs[:, i]
        ax[1, i].plot(csd[0].times, stc, label=csd[j].comment)
ax[1, 1].legend()


csd = read_evokeds(root / "results" / "group_csd-ave.fif")
X = csd[-1].data
X -= X.mean(axis=1, keepdims=True)  # subtract each channels mean
cov = (X @ X.T) / (X.shape[1] - 1)

eig_vals, eig_vecs = np.linalg.eig(cov)
# scale eigenvalues to percent variance explained
eig_vals = (eig_vals / eig_vals.sum()) * 100


fig, ax = plt.subplots(2, 3)
for i in range(3):
    plot_topomap(eig_vecs[:, i], csd[0].info, show=False, axes=ax[0, i])
    ax[0, i].set(title=f"PC{i+1}: {eig_vals[i].round(1)}%")
    for j in range(4):
        X = csd[j].data
        stc = X.T @ eig_vecs[:, i]
        ax[1, i].plot(csd[0].times, stc, label=csd[j].comment)
ax[1, 1].legend()
