from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.transforms as mtransforms
import matplotlib.gridspec as gridspec
from scipy import stats

root = Path(__file__).parent.parent.parent.absolute()
plt.style.use(["science", "no-latex"])
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig = plt.figure()
gs = gridspec.GridSpec(4, 4, figure=fig)
ax0 = plt.subplot(gs[:, :2])
ax1 = plt.subplot(gs[:2, :2])
ax2 = plt.subplot(gs[2:, :2])
ax3 = plt.subplot(gs[:, 2:], projection="3d")


def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return y


# generate opponent-channel and population-rate code
alpha = 50
x = np.linspace(-90, 90, 1000)
y1 = stats.norm.pdf(x, -80, alpha)  # channel 1
y2 = stats.norm.pdf(x, 80, alpha)  # channel 2
y3 = sigmoid(x, L=180, x0=0.5, k=0.05, b=-90)  # population rate-code

# normalize
y1 /= y1.max()
y2 /= y2.max()
y3 += y3.max()
y3 /= y3.max()

# plot the separate codes
ax1.plot(x, y1)
ax1.plot(x, y2)
ax2.plot(x, y3, color=colors[2])
ax1.set(xlabel="Azimuth [\u00b0]", ylabel="Activity [a.u.]")
ax2.set(xlabel="Elevation [\u00b0]", ylabel="Activity [a.u.]")

# Now combine the codes by summing the difference between the opponent channels
# and the population activity across elevation and azimuth
ele, azi = np.arange(-90, 100, 0.1), np.arange(-90, 100, 0.1)
X, Y = np.meshgrid(ele, azi)
Z1 = stats.norm.pdf(Y, -40, alpha) - stats.norm.pdf(Y, 40, alpha)
Z1 += Z1.max()
Z1 /= Z1.max()
Z2 = sigmoid(X, L=180, x0=0.5, k=0.05, b=-90)
Z2 += Z2.max()
Z2 /= Z2.max()
Z = Z1 + Z2
Z /= Z.max()

ax3.plot_surface(X, Y, Z, vmin=Z.min() * 2, cmap=cm.coolwarm)

ax3.set_xlabel("Elevation [\u00b0]")
ax3.set_ylabel("Azimuth [\u00b0]")
ax3.set_zlabel("Activity [a.u.]")
ax3.view_init(-145, 30)
plt.subplots_adjust(left=0.1, right=0.9, hspace=0.7)


fig.text(0.12, 0.82, "A", fontsize=10, weight="bold")
fig.text(0.12, 0.4, "B", fontsize=10, weight="bold")
fig.text(0.55, 0.75, "C", fontsize=10, weight="bold")

plt.savefig(root / "results" / "plots" / "joint_code.png", dpi=300, bbox_inches="tight")
plt.show()
