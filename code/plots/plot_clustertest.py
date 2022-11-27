from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from mne.viz import plot_topomap
from mne.io import read_info

root = Path(__file__).parent.parent()
info = read_info(root / "preprocessed" / "sub-100" / "sub-100-epo.fif")
subfolders = list((root / "results").glob("sub*"))
subfolders.sort()

# TODO: check why this doesn't return significant values
topo = np.zeros(64)
group_F = np.zeros((205, 64))
for subfolder in subfolders[0:23]:  # experiment I
    data = np.load(
        subfolder / f"{subfolder.name}_a+375_cluster.npy", allow_pickle=True
    ).item()
    F, clusters, p = data.values()
    sig_idx = np.where(p < 0.05)[0]
    print(sig_idx)
    if sig_idx.size > 0:
        cluster = np.unique(np.concatenate([clusters[i][0] for i in sig_idx]))
        topo += F[cluster, :].mean(axis=0)
        group_F += F


for subfolder in subfolders[23:]:  # experiment I
    data = np.load(
        subfolder / f"{subfolder.name}_cluster.npy", allow_pickle=True
    ).item()
    F, clusters, p = data.values()
    sig_idx = np.where(p < 0.05)[0]
    print(sig_idx)
    if sig_idx.size > 0:
        cluster = np.unique(np.concatenate([clusters[i][0] for i in sig_idx]))
        cluter = cluster[cluster > 750]
        subject_topo = statistic[cluster, :].mean(axis=0)
