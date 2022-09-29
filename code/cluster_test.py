import sys
from pathlib import Path
import argparse
import numpy as np
from mne.stats import permutation_cluster_test
from mne.channels import find_ch_adjacency
root = Path(__file__).parent.parent.absolute()
sys.path.append(str(root/"code"))
from utils import get_epochs
parser = argparse.ArgumentParser()
parser.add_argument("subject", type=str)
parser.add_argument("n_permutations", type=int, default=1000)
args = parser.parse_args()

epochs = get_epochs(args.subject)
event_ids = list(epochs.event_id.values())
adjacency = find_ch_adjacency(epochs.info, "eeg")[0]
indices = [np.where(
    epochs.events[:, 2] == event_id)[0] for event_id in event_ids]
data = [epochs.get_data()[idx, :, :].transpose(0, 2, 1) for idx in indices]
T_obs, clusters, cluster_p_values, H0 = \
    permutation_cluster_test(data,
                             n_permutations=args.n_permutations, adjacency=adjacency)
dat = {"statistic": T_obs, "clusters": clusters,
        "clusters_p": cluster_p_values}
np.save(
    root/"freefield"/"output"/args.subject/"permutation_test_results.npy", dat)
