import sys
from pathlib import Path
import itertools
import numpy as np
import argparse
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mne.decoding import SlidingEstimator, cross_val_multiscore
root = Path(__file__).parent.parent.absolute()
sys.path.append(str(root/"code"))
from utils import get_epochs, get_transforms, apply_transform

parser = argparse.ArgumentParser()
parser.add_argument('experiment', type=str, choices=["freefield", "headphones"])
parser.add_argument('subject', type=str)
parser.add_argument('-n', '--n_components', type=int, default=3)
parser.add_argument('-cv', '--cross_val', type=int, default=10)
parser.add_argument('--n_jobs', type=int, default=1)
args = parser.parse_args()

epochs = get_epochs(args.subject)
epochs.resample(100)
# to_jd1, from_jd1, to_jd2, from_jd2 = get_transforms(args.subject)
# components = apply_transform(apply_transform(epochs.get_data(), to_jd1), to_jd2[:, 0:args.n_components])
components = epochs._data
events, event_id = epochs.events, epochs.event_id
n_times = len(epochs.times)
del epochs
combinations = [",".join(map(str, comb)) for comb in itertools.combinations(event_id.keys(), 2)]
combinations = [comb.split(",") for comb in combinations]

scores = np.zeros((len(combinations), args.cross_val, n_times))
for i, (con1, con2) in enumerate(combinations):
    mask = np.logical_or(events[:, 2] == event_id[con1], events[:, 2] == event_id[con2])
    X = components[mask]
    y = events[mask][:, 2]
    clf = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs'))
    decoder = SlidingEstimator(clf, n_jobs=args.n_jobs, scoring='roc_auc', verbose=True)
    scores[i, :, :] = cross_val_multiscore(decoder, X, y, cv=args.cross_val, n_jobs=args.n_jobs)

np.save(root/args.experiment/"output"/args.subject/f"combinations.npy", np.array(combinations))
np.save(root/args.experiment/"output"/args.subject/f"accuracy.npy", scores)
