"""Plot examples."""

import json
from argparse import ArgumentParser

from matplotlib import pyplot as plt
import numpy as np

from dart import dataset


def _parse():
    p = ArgumentParser()
    p.add_argument("-p", "--path", help="Path to output base name.")
    return p


if __name__ == '__main__':
    args = _parse().parse_args()

    with open(args.path + ".json") as f:
        cfg = json.load(f)

    y_true = dataset.load_arrays(cfg["dataset"]["path"])["rad"]
    y_pred = dataset.load_arrays(args.path + ".mat")["rad"]

    def _show(ax, im):
        # ax.imshow(np.sqrt(im))
        ax.imshow(im)

    fig, axs = plt.subplots(6, 6, figsize=(16, 16))

    idxs = np.random.choice(y_true.shape[0], 18, replace=False)
    for idx, pair in zip(idxs, axs.reshape(-1, 2)):
        _show(pair[0], y_true[idx])
        _show(pair[1], y_pred[idx])
        pair[0].set_title("actual")
        pair[1].set_title("predicted")

    fig.tight_layout()
    fig.savefig(args.path + ".ex.png")
