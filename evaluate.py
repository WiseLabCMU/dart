"""Evaluate model."""

import os
import json
from tqdm import tqdm
from functools import partial
from argparse import ArgumentParser

import numpy as np
from jax import numpy as jnp
import jax
from scipy.io import savemat

from dart import dataset, DART, VirtualCamera


def _parse():
    p = ArgumentParser()
    p.add_argument("-p", "--path", help="File path to output base name.")
    p.add_argument(
        "-r", "--key", default=42, type=int, help="Random seed.")
    p.add_argument("-b", "--batch", default=32, type=int, help="Batch size")
    p.add_argument(
        "-a", "--all", default=False, action="store_true",
        help="Render all images instead of only the validation set.")
    p.add_argument(
        "-c", "--camera", default=False, action="store_true",
        help="Render camera image instead of radar image.")
    p.add_argument(
        "--clip", default=0.01, type=float,
        help="Inclusion threshold for camera rendering.")
    return p


def _render_dataset(state, args, traj):
    if args.camera:
        _render = jax.jit(partial(
            dart.camera, key=args.key, params=state,
            camera=VirtualCamera(d=256, max_depth=3.2, f=1.0, clip=args.clip)))

        def render(b):
            return _render(batch=b).to_rgb()
    else:
        _render = jax.jit(partial(dart.render, key=args.key, params=state))

        def render(b):
            return np.asarray(_render(batch=b))

    frames = []
    for batch in tqdm(traj.batch(args.batch)):
        frames.append(render(jax.tree_util.tree_map(jnp.array, batch)))
    return {"cam" if args.camera else "rad": np.concatenate(frames, axis=0)}


if __name__ == '__main__':

    args = _parse().parse_args()

    with open(os.path.join(args.path, "metadata.json")) as f:
        cfg = json.load(f)

    dart = DART.from_config(**cfg)
    state = dart.load(os.path.join(args.path, "model.chkpt"))

    if args.all:
        traj = dataset.trajectory(cfg["dataset"]["path"])
    else:
        subset = np.load(os.path.join(args.path, "metadata.npz"))["validx"]
        traj = dataset.trajectory(cfg["dataset"]["path"], subset=subset)

    out = _render_dataset(state, args, traj)

    outfile = "{}{}.mat".format(
        "cam" if args.camera else "pred", "_all" if args.all else "")
    savemat(os.path.join(args.path, outfile), out)
