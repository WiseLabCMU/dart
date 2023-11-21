"""Plot SSIM CDF."""

import os
from matplotlib import pyplot as plt
import numpy as np

from _stats import load_dir


fig, axs = plt.subplots(2, 6, figsize=(12, 4))
datasets = {
    "boxes2": "Lab 1",
    "boxes3": "Lab 2",
    "wiselab4": "Office 1",
    "wiselab5": "Office 2",
    "mallesh-half": "Rowhouse 1",
    "mallesh-1br": "Rowhouse 2",
    "mallesh-full": "Rowhouse 3",
    "agr-ground": "House 1",
    "agr-full": "House 2",
    "agr-yard": "Yard",
    "tianshu-full": "Apartment 1",
    "tianshu-half": "Apartment 2"
}
methods = {
    "ngpsh": ("DART", 'C0', '-'),
    "lidar": ("Lidar", 'C1', '--'),
    "nearest": ("Nearest", 'C2', ':'),
    "cfar": ("CFAR", 'C3', '-.')
}

for (ds, label), ax in zip(datasets.items(), axs.reshape(-1)):
    ssim = load_dir(ds)
    _ref = np.load(os.path.join("data", ds, "baselines", "reference.npz"))
    ref = np.mean(_ref["ssim"], axis=0)

    ax.axvline(
        ref[0], color='black', linestyle='--', label='25/30/35db Reference',
        linewidth=1.0)
    ax.axvline(ref[1], color='black', linestyle='--', linewidth=1.0)
    ax.axvline(ref[2], color='black', linestyle='--', linewidth=1.0)

    for k, (desc, color, ls) in methods.items():
        if k in ssim:
            v = ssim[k]
            ax.plot(
                np.sort(v), np.arange(v.shape[0]) / v.shape[0],
                label=desc, color=color, linestyle=ls)

    ax.grid(visible=True)
    ax.set_xlim(0.35, 0.85)
    ax.text(0.83, 0.01, label, ha='right', va='bottom', backgroundcolor='white')

for ax in axs[:, 1:].reshape(-1):
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
for ax in axs[:-1].reshape(-1):
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)


fig.tight_layout(h_pad=0.2, w_pad=0.5)
axs[-1, -1].legend(
    ncols=5, loc='upper right', bbox_to_anchor=(1.05, -0.15), frameon=False)
axs[1, 0].set_ylabel(r"Cumulative Probability $\longrightarrow$", loc='bottom')
axs[1, 0].set_xlabel(r"SSIM (higher is better) $\longrightarrow$", loc='left')
fig.savefig("figures/ssim.pdf", bbox_inches='tight')
