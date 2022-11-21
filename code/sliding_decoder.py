from pathlib import Path
import itertools
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mne import read_epochs
from mne.decoding import SlidingEstimator, cross_val_multiscore

root = Path(__file__).parent.parent.absolute()
n_crossval = 100

subfolders = list((root / "preprocessed").glob("sub-1*"))

for subfolder in subfolders:
    epochs = read_epochs(subfolder / f"{subfolder.name}-epo.fif")
    events, event_id = epochs.events, epochs.event_id
    n_times = len(epochs.times)
    combinations = [
        ",".join(map(str, c)) for c in itertools.combinations(epochs.event_id.keys(), 2)
    ]
    results = {}
    combinations = [comb.split(",") for comb in combinations]
    scores = np.zeros((len(combinations), n_crossval, n_times))
    for i, (con1, con2) in enumerate(combinations):
        mask = np.logical_or(
            events[:, 2] == event_id[con1], events[:, 2] == event_id[con2]
        )
        X = epochs.get_data()[mask]
        y = events[mask][:, 2]
        clf = make_pipeline(
            StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=1000)
        )
        decoder = SlidingEstimator(clf, scoring="roc_auc", verbose=True)
        scores = cross_val_multiscore(decoder, X, y, cv=n_crossval)
        results[f"{con1} vs {con2}"] = scores
        np.save(
            root / "results" / subfolder.name / f"{subfolder.name}-decoding.npy",
            results,
        )
