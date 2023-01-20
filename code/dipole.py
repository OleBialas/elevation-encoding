from pathlib import Path
from mne import (
    read_evokeds,
    read_cov,
    make_sphere_model,
    fit_dipole,
    make_forward_dipole,
)
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.channels import make_1020_channel_selections

root = Path(__file__).parent.parent.absolute()

t_dipole = 1.45
evokeds = read_evokeds(root / "results" / "grand_averageII_conditions-ave.fif")
grand_average = read_evokeds(root / "results" / "grand_averageII-ave.fif")[0]
bem = make_sphere_model(r0="auto", head_radius="auto", info=evokeds[0].info)

for isub, sub in enumerate((root / "preprocessed").glob("sub-1*")):
    if isub == 0:
        convariance = read_cov(sub / f"{sub.name}_noise-cov.fif")
    else:
        convariance += read_cov(sub / f"{sub.name}_noise-cov.fif")

evoked = evokeds[-1].copy()
evoked.crop(t_dipole, t_dipole)
evoked.set_eeg_reference("average", projection=True)
channels = make_1020_channel_selections(evoked.info)

# fit the dipole to left and right hemisphere at time t_dipole
evoked_right = evoked.copy().pick_channels(
    np.array(evoked.info["ch_names"])[channels["Right"]]
)
covariance_right = covariance.copy().pick_channels(
    np.array(evoked.info["ch_names"])[channels["Right"]]
)
evoked_left = evoked.copy().pick_channels(
    np.array(evoked.info["ch_names"])[channels["Left"]]
)
covariance_left = covariance.copy().pick_channels(
    np.array(evoked.info["ch_names"])[channels["Left"]]
)

dipole_right = fit_dipole(evoked_right, covariance_right, bem)[0]
dipole_left = fit_dipole(evoked_left, covariance_left, bem)[0]


fwd, _ = make_forward_dipole([dipole_left, dipole_right], bem, evoked.info)

inv = make_inverse_operator(evoked.info, fwd, covariance, fixed=True, depth=0)

# use the inverse operator to predict the evoked for each channel
fig, ax = plt.subplots(2)
pred_evokeds = []
for evoked in evokeds:
    evoked.set_eeg_reference("average", projection=True)
    stc = apply_inverse(evoked, inv, method="MNE", lambda2=1e-6)
    for i, y in enumerate(stc.data):
        ax[i].plot(stc.times, y, label=evoked.comment)
    pred_evokeds.append(simulate_evoked(fwd, stc, evoked.info, cov=None, nave=np.inf))
