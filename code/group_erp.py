"""
Compute a linear tuning curve for each subject
"""
import sys
from pathlib import Path
import numpy as np
from mne.channels import make_1020_channel_selections
list_of_subjects = np.loadtxt(
        "/home/ole/projects/elevation/code/list_of_subjects.csv",
        delimiter=",", skiprows=1, dtype=str)[:, 0]
root = Path(__file__).parent.parent.absolute()
sys.path.append(str(root/"code"))
from utils import get_epochs, get_clustertest

channels = ["FT9", "FT10"]
data = np.zeros((2, 30, 4, 1251))
for i_sub, subject in enumerate(list_of_subjects):
    epochs = get_epochs(subject, targets=False)
    epochs.equalize_event_counts()
    for i_ch, channel in enumerate(channels):
        epochs_ch = epochs.copy()
        epochs_ch.pick_channels([channel])
        for ie, event in enumerate(epochs.event_id.keys()):
            data[i_ch, i_sub, ie] = epochs_ch[event].average().data.flatten()
np.save(root/"freefield"/"output"/"group_erp.npy", data)
