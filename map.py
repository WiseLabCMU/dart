"""Draw model map."""

import json
import os
from argparse import ArgumentParser

import numpy as np
from matplotlib import pyplot as plt
from jax import numpy as jnp

from dart import DART


def _parse():
    p = ArgumentParser()
    p.add_argument("-p", "--path", help="File path to output base name.")
    p.add_argument(
        "-r", "--radius", default=0.6, type=float,
        help="Size of area to show.")
    return p


if __name__ == '__main__':

    args = _parse().parse_args()

    with open(os.path.join(args.path, "metadata.json")) as f:
        cfg = json.load(f)

    dart = DART.from_config(**cfg)
    state = dart.load(os.path.join(args.path, "model.chkpt"))

    r = args.radius
    x = jnp.linspace(-r, r, 100)
    y = jnp.linspace(-r, r, 100)
    z = jnp.array([-0.1, 0.0, 0.2, 0.4])
    steps = np.array([0, 1, 2, 3])
    sigma, alpha = dart.grid(state.params, x, y, z)

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    for layer, ax in zip(steps, axs[0]):
        tmp = ax.imshow(sigma[:, :, layer].T, origin='lower')

        ax.set_xticks(np.linspace(0, 100, 5))
        ax.set_yticks(np.linspace(0, 100, 5))
        ax.set_xticklabels(["{:.2f}".format(x) for x in np.linspace(-r, r, 5)])
        ax.set_yticklabels(["{:.2f}".format(x) for x in np.linspace(-r, r, 5)])
        ax.set_title("$\\sigma: z={:.2f}$".format(z[layer]))
        fig.colorbar(tmp)

    for layer, ax in zip(steps, axs[1]):
        tmp = ax.imshow(alpha[:, :, layer].T, origin='lower')

        ax.set_xticks(np.linspace(0, 100, 5))
        ax.set_yticks(np.linspace(0, 100, 5))
        ax.set_xticklabels(["{:.2f}".format(x) for x in np.linspace(-r, r, 5)])
        ax.set_yticklabels(["{:.2f}".format(x) for x in np.linspace(-r, r, 5)])
        ax.set_title("$\\alpha: z={:.2f}$".format(z[layer]))
        fig.colorbar(tmp)

    fig.tight_layout()
    fig.savefig(os.path.join(args.path, "map.png"))
