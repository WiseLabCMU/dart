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
    p.add_argument("-b", "--batch", default=None, type=int, help="Batch size")
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


def _render_camera(state, args, traj):
    render = jax.jit(partial(
        dart.camera, key=args.key, params=state,
        camera=VirtualCamera(d=256, max_depth=3.2, f=1.0, clip=args.clip)))

    d, s, a = [], [], []
    for batch in tqdm(traj.batch(args.batch)):
        res = render(batch=jax.tree_util.tree_map(jnp.array, batch))
        d.append(np.asarray(res.d))
        s.append(np.asarray(res.sigma))
        a.append(np.asarray(res.a))
    return {
        "d": np.concatenate(d, axis=0),
        "sigma": np.concatenate(s, axis=0),
        "a": np.concatenate(a, axis=0)
    }


def _render_radar(state, args, traj):
    render = jax.jit(partial(dart.render, key=args.key, params=state))
    frames = []
    for batch in tqdm(traj.batch(args.batch)):
        frames.append(np.asarray(
            render(batch=jax.tree_util.tree_map(jnp.array, batch))))
    return {"rad": np.concatenate(frames, axis=0)}


if __name__ == '__main__':

    args = _parse().parse_args()

    if args.batch is None:
        args.batch = 4 if args.camera else 32

    with open(os.path.join(args.path, "metadata.json")) as f:
        cfg = json.load(f)

    dart = DART.from_config(**cfg)
    state = dart.load(os.path.join(args.path, "model.chkpt"))

    if args.all:
        traj = dataset.trajectory(cfg["dataset"]["path"])
    else:
        subset = np.load(os.path.join(args.path, "metadata.npz"))["validx"]
        traj = dataset.trajectory(cfg["dataset"]["path"], subset=subset)

    if args.camera:
        out = _render_camera(state, args, traj)
    else:
        out = _render_radar(state, args, traj)

    outfile = "{}{}.mat".format(
        "cam" if args.camera else "pred", "_all" if args.all else "")
    savemat(os.path.join(args.path, outfile), out)
