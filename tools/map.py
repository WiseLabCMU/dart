"""Draw model map."""

import json
import os
from tqdm import tqdm

from scipy.io import savemat
import numpy as np
from jax import numpy as jnp

from dart import DART


_desc = "Evaluate DART model in a grid."


def _parse(p):
    p.add_argument("-p", "--path", help="File path to output base name.")
    p.add_argument(
        "-l", "--lower", nargs='+', type=float, default=[-1.0, -1.0, -1.0],
        help="Lower coordinate in x y z form.")
    p.add_argument(
        "-u", "--upper", nargs='+', type=float, default=[1.0, 1.0, 1.0],
        help="Upper coordinate in x y z form.")
    p.add_argument(
        "-r", "--resolution", nargs='+', type=int, default=[512],
        help="Map resolution; can be nx ny nz for an arbitrary shape, or a "
        "single resolution for a cube.")
    p.add_argument(
        "-b", "--batch", type=int, default=16, help="Batch size along the z "
        "axis for breaking up high resolution grids.")
    return p


def _main(args):
    assert len(args.lower) == 3
    assert len(args.upper) == 3
    args.resolution = args.resolution * 3

    with open(os.path.join(args.path, "metadata.json")) as f:
        cfg = json.load(f)

    dart = DART.from_config(**cfg)
    state = dart.load(os.path.join(args.path, "model.chkpt"))

    x, y, z = [
        jnp.linspace(lower, upper, res) for lower, upper, res in
        zip(args.lower, args.upper, args.resolution)]

    sigma, alpha = [], []
    for _ in tqdm(range(int(np.ceil(args.resolution[2] / args.batch)))):
        _sigma, _alpha = dart.grid(state.params, x, y, z[:args.batch])
        z = z[args.batch:]
        sigma.append(_sigma)
        alpha.append(_alpha)
    sigma = np.concatenate(sigma, axis=2)
    alpha = np.concatenate(alpha, axis=2)

    savemat(
        os.path.join(args.path, "map.mat"), {"sigma": sigma, "alpha": alpha})
