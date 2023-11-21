import numpy as np
import h5py
from matplotlib import pyplot as plt

mapfile = h5py.File("results/tianshu-full/ngpsh/map.h5")
sigma = mapfile["sigma"]

fig, axs = plt.subplots(1, 1, figsize=(16, 10))
t = np.mean(sigma[:, :, 135:145], axis=2)
lower, upper = np.percentile(t, [50, 98])
axs.imshow(np.clip(t, lower, upper), cmap='inferno')
axs.set_xticks([])
axs.set_yticks([])
fig.tight_layout(pad=0.0)
fig.savefig("figures/hero_floorplan.png", dpi=300, bbox_inches='tight')
 