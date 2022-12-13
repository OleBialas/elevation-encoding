from pathlib import Path
import re
import numpy as np
from mne import read_epochs
from mne.stats import permutation_cluster_test
from mne.channels import find_ch_adjacency

root = Path(__file__).parent.parent.absolute()
n_permutations = 10000


def run_cluster_test(epochs, event_ids):
    """Run a cluster test on `epochs` comparing the conditions in `event_ids`."""
    # get the trials corresponding to the different stimuli
    conditions = list(epochs.event_id.keys())[:-1]
    epochs.equalize_event_counts(conditions)
    adjacency = find_ch_adjacency(epochs.info, "eeg")[0]
    indices = [np.where(epochs.events[:, 2] == event_id)[0] for event_id in event_ids]
    data = [epochs.get_data()[idx, :, :].transpose(0, 2, 1) for idx in indices]
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
        data, n_permutations=n_permutations, adjacency=adjacency
    )
    return {"statistic": T_obs, "clusters": clusters, "clusters_p": cluster_p_values}


for subfolder in (root / "preprocessed").glob("sub-*"):
    epochs = read_epochs(subfolder / f"{subfolder.name}-epo.fif")
    outfolder = root / "results" / subfolder.name
    if not outfolder.exists():
        outfolder.mkdir()
    if int(re.search(r"\d+", subfolder.name).group()) < 100:
        event_ids = [4, 5, 6, 7, 8, 9]
    else:  # for experiment II, just run a single cluster test
        event_ids = list(epochs.event_id.values())
    result = run_cluster_test(epochs, event_ids)
    np.save(outfolder / f"{subfolder.name}_cluster.npy", result)
