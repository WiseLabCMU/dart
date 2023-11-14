"""Evaluate DART model in a grid."""

import os
import h5py
from tqdm import tqdm
from functools import partial

import numpy as np
from jax import numpy as jnp
import jax

from dart import DartResult, DART


def _parse(p):
    p.add_argument("-p", "--path", help="File path to output base name.")
    p.add_argument("-c", "--checkpoint", help="Load specific checkpoint.")
    p.add_argument(
        "-l", "--lower", nargs='+', type=float, default=None,
        help="Lower coordinate in x y z form.")
    p.add_argument(
        "-u", "--upper", nargs='+', type=float, default=None,
        help="Upper coordinate in x y z form.")
    p.add_argument(
        "--padding", type=float, nargs='+', default=[4.0, 4.0, 2.0],
        help="Region padding relative to trajectory min/max.")
    p.add_argument(
        "-r", "--resolution", type=int, default=25,
        help="Map resolution, in units per meter.")
    p.add_argument(
        "-b", "--batch", type=int, default=4, help="Batch size along the z "
        "axis for breaking up high resolution grids.")
    return p


def _set_bounds(args, res):
    if args.lower is None or args.upper is None:
        args.padding = np.array((args.padding * 3)[:3])
        x = np.array(h5py.File(
            os.path.join(res.DATADIR, "trajectory.h5"))["pos"])
        args.lower = np.min(x, axis=0) - args.padding
        args.upper = np.max(x, axis=0) + args.padding
    else:
        assert len(args.lower) == 3
        assert len(args.upper) == 3
        args.lower = np.array(args.lower)
        args.upper = np.array(args.upper)


def _main(args):
    result = DartResult(args.path)
    _set_bounds(args, result)

    resolution = (args.resolution * (args.upper - args.lower)).astype(int)
    print("Bounds: {:.1f}x{:.1f}x{:.1f}m ({}x{}x{}px)".format(
        *(args.upper - args.lower), *resolution))

    if args.checkpoint is None:
        outfile = "map.h5"
        args.checkpoint = "model"
    else:
        outfile = "map.{}.h5".format(args.checkpoint)
        args.checkpoint = os.path.join("checkpoints", args.checkpoint)

    dart = DART.from_file(args.path)
    params = dart.load(os.path.join(args.path, args.checkpoint))

    x, y, z = [
        jnp.linspace(lower, upper, res) for lower, upper, res in
        zip(args.lower, args.upper, resolution)]

    render = jax.jit(partial(dart.grid, params, x, y))

    sigma, alpha = [], []
    for _ in tqdm(range(int(np.ceil(resolution[2] / args.batch)))):
        _sigma, _alpha = render(z=z[:args.batch])
        z = z[args.batch:]
        sigma.append(_sigma)
        alpha.append(_alpha)
    sigma = np.concatenate(sigma, axis=2)
    alpha = np.concatenate(alpha, axis=2)

    result.save(outfile, {
        "sigma": sigma, "alpha": alpha,
        "lower": args.lower, "upper": args.upper
    })
