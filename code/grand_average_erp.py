"""
Compute a linear tuning curve for each subject
"""
import re
from pathlib import Path
from mne import grand_average, read_epochs

root = Path(__file__).parent.parent.absolute()

evokeds1, evokeds2 = [], []
for subfolder in (root / "preprocessed").glob("sub-*"):
    epochs = read_epochs(subfolder / f"{subfolder.name}-epo.fif")
    if int(re.search(r"\d+", subfolder.name).group()) < 100:
        evokeds1.append(epochs.average())
    else:
        evokeds2.append(epochs.average())
evoked1 = grand_average(evokeds1)
evoked2 = grand_average(evokeds2)
# evoked2.info["bads"].append("F1")
evoked2.interpolate_bads()

evoked1.save(root / "results" / "grand_averageI-evo.fif", overwrite=True)
evoked2.save(root / "results" / "grand_averageII-evo.fif", overwrite=True)
