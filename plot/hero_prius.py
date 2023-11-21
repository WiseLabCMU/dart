import numpy as np
import h5py
from matplotlib import pyplot as plt

mapfile = h5py.File("results/prius/ngpsh/map.h5")
sigma = np.swapaxes(mapfile["sigma"], 0, 2)
sigma = np.swapaxes(sigma, 1, 2)
sigma = np.flip(sigma, axis=1)

fig, axs = plt.subplots(1, 1, figsize=(16, 10))
t = np.sqrt(np.mean(sigma[120:170, 275:475, 320:600], axis=0))
lower, upper = np.percentile(t, [50, 98])
axs.imshow(np.clip(t, lower, upper), cmap='inferno')
axs.set_xticks([])
axs.set_yticks([])
fig.tight_layout(pad=0.0)
fig.savefig("figures/hero_prius.png", dpi=300, bbox_inches='tight')
