"""Plot examples."""

import json
import os
from argparse import ArgumentParser

from matplotlib import pyplot as plt
import numpy as np

from dart import dataset


def _parse():
    p = ArgumentParser()
    p.add_argument("-p", "--path", help="Path to output base name.")
    p.add_argument(
        "-k", "--key", type=int, default=42,
        help="Random seed for choosing samples to plot.")
    p.add_argument(
        "-a", "--all", default=False, action="store_true",
        help="Render all images instead of only the validation set.")
    return p


if __name__ == '__main__':
    args = _parse().parse_args()

    with open(os.path.join(args.path, "metadata.json")) as f:
        cfg = json.load(f)

    y_true = dataset.load_arrays(cfg["dataset"]["path"])["rad"]
    if args.all:
        y_pred_file = "pred_all.mat"
    else:
        y_pred_file = "pred.mat"
        validx = np.load(os.path.join(args.path, "metadata.npz"))["validx"]
        y_true = y_true[validx]
    y_pred = dataset.load_arrays(os.path.join(args.path, y_pred_file))["rad"]

    y_true = y_true[:, :y_pred.shape[1], 32:-32] / cfg["dataset"]["norm"]

    rng = np.random.default_rng(args.key)
    idxs = np.sort(rng.choice(y_true.shape[0], 18, replace=False))

    fig = plt.figure(figsize=(16, 16))
    gridspecs = fig.add_gridspec(6, 3, hspace=0.03, wspace=0.03)
    for idx, gs in zip(idxs, gridspecs):
        pair = gs.subgridspec(1, 2, wspace=0.0).subplots()
        vmin = min(np.min(y_true[idx]), np.min(y_pred[idx]))
        vmax = max(np.max(y_true[idx]), np.max(y_pred[idx]))
        pair[0].imshow(y_true[idx], vmin=vmin, vmax=vmax)
        pair[1].imshow(y_pred[idx], vmin=vmin, vmax=vmax)
        pair[0].text(1, 5, "#{} Measured".format(idx), color='white')
        pair[1].text(1, 5, "#{} Predicted".format(idx), color='white')

        for ax in pair:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.savefig(
        os.path.join(args.path, "pred_all.png" if args.all else "pred.png"),
        bbox_inches='tight', pad_inches=0.2, dpi=200)
