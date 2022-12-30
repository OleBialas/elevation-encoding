"""
For both experiments, get each subjects cluster with the largest cumulative F-value
If the cluster was significant, sum F-scores in that cluster for every channel and
take the z-score to get the topographical ditribution of the effect. Average all the
significant topographies. For each experiment, write a .csv table with each subject's
largest cluster score and the corresponding p-value as well as a .npy file with the
average topography
"""

from pathlib import Path
import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt
from mne import read_epochs

root = Path(__file__).parent.parent.absolute()
epochs = read_epochs(root / "preprocessed" / "sub-100" / "sub-100-epo.fif")
info, times = epochs.info, epochs.times
del epochs
subfolders = list((root / "results").glob("sub*"))
subfolders.sort()

# subject ID, cumulative F-score, p-value, start and stop, of each subjet's largest cluster
clustersI = np.zeros((23, 5))
clustersII = np.zeros((30, 5))
# Topographical distribution of significant clusters
topoI = np.zeros(64)
topoII = np.zeros(64)

for isub, subfolder in enumerate(subfolders):  # experiment I
    sub_id = int(subfolder.name.split("-")[1])
    data = np.load(
        subfolder / f"{subfolder.name}_cluster.npy", allow_pickle=True
    ).item()
    F, clusters, p, _, _ = data.values()
    mass = np.zeros(len(clusters))
    for ic, c in enumerate(clusters):  # get the mass of each cluster
        mass[ic] = F[c].sum()
    idx = np.argmax(mass)
    tmin, tmax = times[clusters[idx][0].min()], times[clusters[idx][0].max()]
    if p[idx] < 0.05:
        # calculate each channel's cumulative F-value in the significant cluster
        topo = F[np.unique(clusters[idx][0]), :].sum(axis=0)
        topo = (topo - topo.mean()) / topo.std()  # zscore
    else:
        topo = np.zeros(64)
    if isub < 23:
        clustersI[isub] = (sub_id, mass[idx], p[idx], tmin, tmax)
        topoI += topo
    else:
        clustersII[isub - 23] = (sub_id, mass[idx], p[idx], tmin, tmax)
        topoII += topo

# generate latex tables
clustersI[:, 1] = clustersI[:, 1].round(2)
clustersI[:, 2] = clustersI[:, 2].round(3)
clustersI[:, 3] = clustersI[:, 3].round(3) - 0.6
clustersI[:, 4] = clustersI[:, 4].round(3) - 0.6
print(
    tabulate(
        clustersI,
        tablefmt="latex",
        headers=("Subject ID", "F-score", "p-value", "Start", "Stop"),
    )
)

clustersII[:, 1] = clustersII[:, 1].round(2)
clustersII[:, 2] = clustersII[:, 2].round(3)
clustersII[:, 3] = clustersII[:, 3].round(3) - 1
clustersII[:, 4] = clustersII[:, 4].round(3) - 1
print(
    tabulate(
        clustersII,
        tablefmt="latex",
        headers=("Subject ID", "F-score", "p-value", "Start", "Stop"),
    )
)


# normalize the topo-plots
topoI /= (clustersI[:, 1] < 0.05).sum()
topoII /= (clustersII[:, 1] < 0.05).sum()

np.save(root / "results" / "ftopoI.npy", topoI)
np.save(root / "results" / "ftopoII.npy", topoII)
np.savetxt(root / "results" / "clustersI.csv", clustersI, header="f,p,tmin,tmax")
np.savetxt(root / "results" / "clustersII.csv", clustersII)
