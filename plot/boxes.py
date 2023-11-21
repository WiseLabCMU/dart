from matplotlib import pyplot as plt
import h5py
import numpy as np


def _im(ax, x):
    lower, upper = np.percentile(x, [0, 99])
    return ax.imshow(np.clip(x, lower, upper), cmap='inferno')

h5 = h5py.File("results/boxes3/ngpsh/map.h5")
fig, axs = plt.subplots(1, 2, figsize=(6, 4))
im1 = _im(axs[0], np.mean(h5['sigma'][210:390, 230:410, 95:130], axis=2))
im2 = _im(axs[1], np.mean(-h5['alpha'][210:390, 230:410, 95:130], axis=2))

axs[0].text(5, 5, "Reflectance", color='white', ha='left', va='top')
axs[1].text(5, 5, "Transmittance", color='white', ha='left', va='top')
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(50, 75, "(1)", color='white')
    ax.text(125, 75, "(2)", color='white')
    ax.text(20, 145, "(5)", color='white')
    ax.text(78, 172, "(4)", color='white')
    ax.text(135, 140, "(3)", color='white')
fig.tight_layout(pad=1.0)

cbar_ax = fig.add_axes([0.021, 0.08, 0.956, 0.04])
fig.colorbar(im2, cax=cbar_ax, orientation='horizontal')
cbar_ax.set_xticks([])
cbar_ax.set_xlabel(r"Increasing Reflectance $\longrightarrow$", loc='left')

cbar_ax2 = cbar_ax.twiny()
cbar_ax2.xaxis.set_ticks_position('bottom')
cbar_ax2.xaxis.set_label_position('bottom')
cbar_ax2.set_xticks([])
cbar_ax2.set_xlabel(r"$\longleftarrow$ Increasing Transmittance", loc='right')

fig.savefig("figures/boxes.pdf", bbox_inches='tight')
