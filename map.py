"""Draw model map."""

import json
from argparse import ArgumentParser

import numpy as np
from matplotlib import pyplot as plt
from jax import numpy as jnp

from dart import DART


def _parse():
    p = ArgumentParser()
    p.add_argument("-p", "--path", help="File path to output base name.")
    return p


if __name__ == '__main__':

    args = _parse().parse_args()

    with open(args.path + ".json") as f:
        cfg = json.load(f)

    dart = DART.from_config(**cfg)
    state = dart.load(args.path + ".chkpt")

    r = 4
    x = jnp.linspace(-r, r, 100)
    y = jnp.linspace(-r, r, 100)
    z = jnp.array([0.0, 0.1, 0.2, 0.3])
    steps = np.array([0, 1, 2, 3])
    grid = dart.grid(state.params, x, y, z)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for layer, ax in zip(steps, axs):
        ax.imshow(grid[:, :, layer, 0].T, origin='lower')

        ax.set_xticks(np.linspace(0, 100, 6))
        ax.set_yticks(np.linspace(0, 100, 6))
        ax.set_xticklabels(["{:.1f}".format(x) for x in np.linspace(-r, r, 6)])
        ax.set_yticklabels(["{:.1f}".format(x) for x in np.linspace(-r, r, 6)])

    fig.savefig(args.path + "_map.png")
