from pathlib import Path
import json
import numpy as np
from mne import events_from_annotations
from mne.io import read_raw_brainvision
from mne.epochs import Epochs
from autoreject import Ransac, AutoReject
from mne.preprocessing import ICA, read_ica, corrmap
from meegkit.dss import dss_line

# tmin, tmax and event_ids for both experiments
epoch_parameters_epx1 = [
    -0.1,
    1.5,
    {
        "37.5": 1,
        "12.5": 2,
        "-12.5": 3,
        "-37.5": 4,
    },
]
epoch_parameters_epx2 = [
    -0.1,
    1.5,
    {
        "37.5 to 12.5": 2,
        "37.5 to -12.5": 3,
        "37.5 to -37.5": 4,
        "-37.5 to 37.5": 5,
        "-37.5 to 12.5": 6,
        "-37.5 to -12.5": 7,
        "deviant": 8,
    },
]

root = Path(__file__).parent.parent.absolute()
for subfolder in (root / "bids").glob("sub*"):
    if int(subfolder.name[-3:]) < 100:
        raw = read_raw_brainvision(
            subfolder / "eeg" / f"{subfolder.name}_task-deviantdetection_eeg.vhdr"
        )
        tmin, tmax, event_ids = epoch_parameters_epx1
    else:
        raw = read_raw_brainvision(
            subfolder / "eeg" / f"{subfolder.name}_task-oneback_eeg.vhdr"
        )
        tmin, tmax, event_ids = epoch_parameters_epx2
    raw.load_data()
    raw.set_montage("standard_1020")
    events = events_from_annotations(raw)[0]

    # STEP 1: Remove power line noise and apply minimum-phase highpass filter
    X = raw.get_data().T
    X, _ = dss_line(X, fline=50, sfreq=raw.info["sfreq"], nremove=3)
    raw._data = X.T  # put the data back into raw
    del X
    raw = raw.filter(l_freq=1, h_freq=None, phase="minimum phase")

    # STEP 2: Epoch and downsample the data
    epochs = Epochs(
        raw, events, event_id=event_ids, tmin=tmin, tmax=tmax, baseline=None
    )
    epochs.resample(128)

    # STEP 3: Apply robust average reference
    r = Ransac(n_jobs=4)
    epochs_interp = r.fit_transform(epochs)
    epochs_interp.set_eeg_reference("average", projection=True)
    epochs.add_proj(epochs_interp.info["projs"])
    epochs.apply_proj()
    del epochs_interp

    # STEP 4: Blink rejection with ICA
    reference = read_ica(root / "code" / "reference_ica.fif")
    component = reference.labels_["blinks"]
    ica = ICA(n_components=0.999, method="fastica")
    ica.fit(epochs)
    ica.labels_["blinks"] = []
    corrmap(
        [reference, ica],
        template=(0, component[0]),
        label="blinks",
        plot=False,
        threshold=0.75,
    )
    ica.apply(epochs, exclude=ica.labels_["blinks"])

    # STEP 5: Remove post-target trials
    if 
