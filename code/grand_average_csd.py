from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from mne import grand_average, read_epochs, write_evokeds
from mne.preprocessing import compute_current_source_density

root = Path(__file__).parent.parent.absolute()
subfolders = list((root / "preprocessed").glob("sub-1*"))

csds = []
for sub in subfolders:
    epochs = read_epochs(sub / f"{sub.name}-epo.fif")
    epochs = epochs.crop(-0.1, 2)
    csd = compute_current_source_density(epochs)
    csds.append(csd.average(by_event_type=True))

csd_grand_average = grand_average([c for csd in csds for c in csd])
csd_conditions = [grand_average([c[i] for c in csds]) for i in range(4)]
conditions = list(epochs.event_id.keys())
for i in range(4):
    csd_conditions[i].comment = conditions[i]
    csd_conditions.append(csd_grand_average)
write_evokeds(root / "results" / "group_csd-ave.fif", csd_conditions)
