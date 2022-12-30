from pathlib import Path
import numpy as np
from scipy.stats import linregress

root = Path(__file__).parent.parent.absolute()
subjects = list((root / "bids").glob("sub*"))
dev_detection, loc_expI, loc_expII, one_back = [], [], [], []


def convna(x):
    if x == b"n/a":
        return np.nan
    else:
        return float(x)
    return


for sub in subjects:
    if int(sub.name.split("-")[1]) < 100:
        task = np.loadtxt(
            sub / "beh" / f"{sub.name}_task-deviantdetection_beh.tsv",
            skiprows=1,
            dtype=int,
        )
        dev_detection.append(task[:, 1].sum() / len(task))
        try:
            test = np.loadtxt(
                sub / "beh" / f"{sub.name}_task-loctest_beh.tsv",
                skiprows=1,
                dtype=float,
            )
            loc_expI.append(linregress(test[:, 0], test[:, 1])[0])
        except FileNotFoundError:
            pass
    else:
        task = np.genfromtxt(
            sub / "beh" / f"{sub.name}_task-oneback_beh.tsv",
            skip_header=1,
            usecols=(1, 2),
        )
        task = task[~np.isnan(task[:, 1])]
        one_back.append(linregress(task[:, 0], task[:, 1])[0])
        test = np.genfromtxt(
            sub / "beh" / f"{sub.name}_task-loctest_beh.tsv",
            skip_header=1,
        )
        test = test[~np.isnan(test[:, 1])]
        loc_expII.append(linregress(test[:, 0], test[:, 1])[0])


print(
    f"experiment I localization test: \n mean EG: {np.mean(loc_expI)} \n standard deviation: {np.std(loc_expI)}"
)

print(
    f"experiment II localization test: \n mean EG: {np.mean(loc_expII)} \n standard deviation: {np.std(loc_expII)}"
)

print(
    f"one-back task: \n mean EG: {np.mean(one_back)} \n standard deviation: {np.std(one_back)}"
)

print(
    f"deviant detections task: \n mean hit rate: {np.mean(dev_detection)} \n standard deviation: {np.std(dev_detection)}"
)
