"""Plot example range-doppler images."""

import numpy as np
from matplotlib import pyplot as plt
from _result import DartResult
from PIL import Image


def _load_images(result, ii, jj):
    gt = result["data/radar.h5"]
    mask = result["data/trajectory.h5"]["mask"]
    lidar = result["data/baselines/lidar.h5"]
    cfar = result["data/baselines/cfar.h5"]
    ours = result["result/rad.h5"]
    traj = result["data/trajectory.h5"]
    val = result["result/metadata.npz"]['val']
    idxs = result["data/data.h5"]["frame_idx"][val]
    test_pose = np.concatenate([traj["pos"][ii], traj["vel"][ii]])
    train_pose = np.concatenate(
        [traj["pos"][:idxs[0] - 1], traj["vel"][:idxs[0] - 1]], axis=1)
    distances = np.sum(np.square(train_pose - test_pose[None, :]), axis=1)
    nearest = gt['rad'][mask][np.argmin(distances)]

    return {
        "gt": gt['rad'][mask][ii, :, :, jj],
        "dart": ours["rad"][ii, :, :, jj],
        "lidar": lidar["rad"][ii, :, :, jj],
        "nearest": nearest[:, :, jj],
        "cfar": cfar["rad"][ii, :, :, jj]
    }


boxes2 = _load_images(DartResult("results/boxes2/ngpsh"), 3763, 5)
wiselab4 = _load_images(DartResult("results/wiselab4/ngpsh"), 12426, 2)
labels = {
    "gt": "Ground Truth",
    "dart": "DART",
    "lidar": "Lidar",
    "nearest": "Nearest",
    "cfar": "CFAR"
}

def _plotrad(ax, img):
    lower, upper = np.percentile(img, [1, 99.9])
    ax.imshow(np.clip(img, lower, upper), aspect='auto')

fig, axs = plt.subplots(2, 6, figsize=(12, 4))

for examples, row in zip([boxes2, wiselab4], axs):
    for ax, (k, v) in zip(row, examples.items()):
        _plotrad(ax, v)
        ax.text(
            0.98, 0.97, labels[k], color='white', ha='right', va='top',
            transform=ax.transAxes)
        ax.set_xticks([64, 128, 196])
        ax.set_yticks([32, 64, 96])
        ax.grid(visible=True, alpha=0.5)

for ax in axs.reshape(-1):
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)

axs[0, -1].text(
    0.98, 0.97, "Lab 1", color='black', ha='right', va='top',
    transform=axs[0, -1].transAxes, backgroundcolor='white')
axs[1, -1].text(
    0.98, 0.97, "Office", color='black', ha='right', va='top',
    transform=axs[1, -1].transAxes, backgroundcolor='white')

axs[-1, 0].set_xlabel(r"Increasing Doppler $\rightarrow$", loc='left')
axs[0, 0].set_ylabel(r"$\leftarrow$ Increasing Range", loc='top')
axs[0, -1].imshow(Image.open("figures/boxes.jpg"), aspect='auto')
axs[1, -1].imshow(Image.open("figures/pillars.jpg"), aspect='auto')


fig.tight_layout()
fig.savefig("figures/examples.pdf", bbox_inches='tight', dpi=300)
