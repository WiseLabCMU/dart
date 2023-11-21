import numpy as np
import h5py
from matplotlib import pyplot as plt

mapfile = h5py.File("results/garden/ngpsh/map.h5")
sigma = np.swapaxes(mapfile["sigma"], 0, 2)
sigma = np.swapaxes(sigma, 1, 2)
sigma = np.flip(sigma, axis=1)

lower, upper = np.percentile(sigma, [5, 99])
sigma = np.clip(sigma, lower, upper)

fig, axs = plt.subplots(1, 1, figsize=(4, 4))
axs.imshow(
    np.mean(sigma[80:90, :, :], axis=0).T, cmap='inferno')
axs.set_xticks([])
axs.set_yticks([])
fig.tight_layout(pad=0.0)
fig.savefig("figures/hero_garden.png", dpi=600, bbox_inches='tight')
