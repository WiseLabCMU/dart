"""Evaluate model."""

import os
from tqdm import tqdm
from functools import partial

import numpy as np
from jax import numpy as jnp
import jax

from dart import VirtualCamera, DartResult


_desc = "Evaluate DART trained checkpoint for an input trajectory."


def _parse(p):
    p.add_argument("-p", "--path", help="File path to output base name.")
    p.add_argument(
        "-r", "--key", default=42, type=int, help="Random seed.")
    p.add_argument(
        "-b", "--batch", default=None, type=int,
        help="Batch size; defaults (4 cam / 32 radar) use 24GB of VRAM.")
    p.add_argument(
        "-a", "--all", default=False, action="store_true",
        help="Render all images instead of only the validation set.")
    p.add_argument(
        "-c", "--camera", default=False, action="store_true",
        help="Render camera image instead of radar image.")
    p.add_argument(
        "--clip", default=0.1, type=float,
        help="Inclusion threshold for camera rendering.")
    p.add_argument(
        "--depth", default=5.0, type=float, help="Maximum depth to render.")
    return p


def _render_camera(dart, params, args, traj):
    render = jax.jit(partial(
        dart.camera, key=args.key, params=params,
        camera=VirtualCamera(
            d=128, max_depth=args.depth, f=1.0, size=(1.0, 1.0),
            res=(128, 128), clip=args.clip)))

    d, s, a = [], [], []
    for batch in tqdm(traj.batch(args.batch)):
        res = render(batch=jax.tree_util.tree_map(jnp.array, batch))
        d.append(np.asarray(res.d, dtype=np.float16))
        s.append(np.asarray(res.sigma, dtype=np.float16))
        a.append(np.asarray(res.a, dtype=np.float16))
    return {
        "d": np.concatenate(d, axis=0),
        "sigma": np.concatenate(s, axis=0),
        "a": np.concatenate(a, axis=0)
    }


def _render_radar(dart, params, args, traj):
    render = jax.jit(partial(dart.render, key=args.key, params=params))
    frames = []
    for batch in tqdm(traj.batch(args.batch)):
        frames.append(np.asarray(
            render(batch=jax.tree_util.tree_map(jnp.array, batch)),
            dtype=np.float16))
    return {"rad": np.concatenate(frames, axis=0)}


def _main(args):

    if args.batch is None:
        args.batch = 4 if args.camera else 32

    result = DartResult(args.path)
    dart = result.dart()
    params = dart.load(os.path.join(args.path, "model"))

    subset = None if args.all else np.load(
        os.path.join(args.path, "metadata.npz"))["validx"]
    traj = result.trajectory_dataset(subset=subset)

    render_func = _render_camera if args.camera else _render_radar
    out = render_func(dart, params, args, traj)

    outfile = DartResult.CAMERA if args.camera else DartResult.RADAR
    if not args.all:
        outfile.replace(".h5", "_val.h5")

    result.save(outfile, out)
