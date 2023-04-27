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
    z = jnp.array([0.0, 0.2, 0.4, 0.8])
    steps = np.array([0, 1, 2, 3])
    sigma, alpha = dart.grid(state.params, x, y, z)

    s_args = {"origin": "lower", "vmin": np.min(sigma), "vmax": np.max(sigma)}
    a_args = {"origin": "lower", "vmin": np.min(alpha), "vmax": np.max(alpha)}

    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    for iz, col in zip(steps, axs.T):
        col[0].imshow(sigma[:, :, iz].T, **s_args)
        col[1].imshow(alpha[:, :, iz].T, **a_args)
        col[0].text(1, 95, "$\\sigma: z={:.1f}$".format(z[iz]), color='white')
        col[1].text(1, 95, "$\\alpha: z={:.1f}$".format(z[iz]), color='white')

    for ax in axs.reshape(-1):
        ax.set_xticks(np.linspace(0, 100, 5))
        ax.set_yticks(np.linspace(0, 100, 5))
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.tick_params(axis=u'both', which=u'both', length=0)
        ax.grid()

    _labels = ["{:.1f}".format(x) for x in np.linspace(-r, r, 5)[1:-1]]
    labels = [""] + _labels + [""]
    for ax in axs[:, 0]:
        ax.set_yticklabels(labels)
    for ax in axs[1]:
        ax.set_xticklabels(labels)

    fig.tight_layout(pad=1.0)
    fig.savefig(
        os.path.join(args.path, "map.png"), bbox_inches='tight',
        pad_inches=0.1, dpi=200)
