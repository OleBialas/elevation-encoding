"""
Compute the grand average evoked response (i.e the average across subjects).
For both experiments, two files are saved - one with the average across
all conditions and one with the average of each condition.
"""
import re
from pathlib import Path
from mne import grand_average, read_epochs, write_evokeds

root = Path(__file__).parent.parent.absolute()

evokeds1, evokeds2 = [], []
evoked1_conditions = {
    "37.5 to 12.5": [],
    "37.5 to -12.5": [],
    "37.5 to -37.5": [],
    "-37.5 to 37.5": [],
    "-37.5 to 12.5": [],
    "-37.5 to -12.5": [],
}
evoked2_conditions = {"37.5": [], "12.5": [], "-12.5": [], "-37.5": []}
for subfolder in (root / "preprocessed").glob("sub-*"):
    epochs = read_epochs(subfolder / f"{subfolder.name}-epo.fif")
    if int(re.search(r"\d+", subfolder.name).group()) < 100:
        evokeds1.append(epochs.average())
        for key in evoked1_conditions.keys():
            evoked1_conditions[key].append(epochs[key].average())
    else:
        evokeds2.append(epochs.average())
        for key in evoked2_conditions.keys():
            evoked2_conditions[key].append(epochs[key].average())
evoked1 = grand_average(evokeds1)
evoked2 = grand_average(evokeds2)

# compute the grand average for each condition
for key in evoked1_conditions.keys():
    evoked1_conditions[key] = grand_average(evoked1_conditions[key])
    evoked1_conditions[key].comment = key
evoked1_conditions = list(evoked1_conditions.values())
for key in evoked2_conditions.keys():
    evoked2_conditions[key] = grand_average(evoked2_conditions[key])
    evoked2_conditions[key].comment = key
evoked2_conditions = list(evoked2_conditions.values())

evoked1.save(root / "results" / "grand_averageI-ave.fif", overwrite=True)
evoked2.save(root / "results" / "grand_averageII-ave.fif", overwrite=True)
write_evokeds(
    root / "results" / "grand_averageII_conditions-ave.fif",
    evoked2_conditions,
    overwrite=True,
)
write_evokeds(
    root / "results" / "grand_averageI_conditions-ave.fif",
    evoked1_conditions,
    overwrite=True,
)
