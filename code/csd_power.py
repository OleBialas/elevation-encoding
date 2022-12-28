from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from mne import read_epochs, grand_average
from mne.preprocessing import compute_current_source_density
from mne.time_frequency import psd_welch
from mne.channels import find_ch_adjacency
from mne.stats import permutation_cluster_test

root = Path(__file__).parent.parent.absolute()
tmin, tmax = 1, None
ttopo = 1.4
vmin, vmax = -1.9, 1.9  # color_scaling

csd_evokeds = [[], [], [], []]
psd_means = [[], [], [], []]
for isub, subfolder in enumerate((root / "preprocessed").glob("sub-1*")):
    epochs = read_epochs(subfolder / f"{subfolder.name}-epo.fif")
    epochs.crop(tmin, tmax)
    csd = compute_current_source_density(epochs)
    if isub == 0:
        keys = [key for key in epochs.event_id.keys()]
        adjacency = find_ch_adjacency(epochs.info, ch_type="eeg")[0]
    psds = []
    for i, key in enumerate(keys):
        csd_evokeds[i].append(csd[key].average())
        psd, freqs = psd_welch(csd[key], n_fft=128, n_jobs=4)

csd_evokeds = [grand_average(csd) for csd in csd_evokeds]
psd_means = [np.stack(psd).transpose(0, 2, 1) for psd in psd_means]

results = permutation_cluster_test(psd_means, adjacency=adjacency)

fig, ax = plt.subplot_mosaic([["A", "B", "C", "D"], ["E", "E", "E", "E"]])

for csd, label, key in zip(csd_evokeds, ["A", "B", "C", "D"], keys):
    csd.plot_topomap(
        ttopo, axes=ax[label], show=False, colorbar=False, vmin=vmin, vmax=vmax
    )
    ax[label].set(title=key)

for i, psd in enumerate(psd_means):
    ax["E"].plot(freqs, psd.mean(axis=0), label=keys[i])
