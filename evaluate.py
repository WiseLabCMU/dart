"""Evaluate model."""

import json
from tqdm import tqdm
from argparse import ArgumentParser

import numpy as np
from jax import numpy as jnp
import jax
from scipy.io import savemat

from dart import dataset, DART


def _parse():
    p = ArgumentParser()
    p.add_argument("-p", "--path", help="File path to output base name.")
    p.add_argument(
        "-r", "--key", default=42, type=int, help="Random seed.")
    p.add_argument("-b", "--batch", default=32, type=int, help="Batch size")
    return p


if __name__ == '__main__':

    args = _parse().parse_args()

    with open(args.path + ".json") as f:
        cfg = json.load(f)

    dart = DART.from_config(**cfg)
    state = dart.load(args.path + ".chkpt")

    traj = dataset.trajectory(cfg["dataset"]["path"])

    root_key = jax.random.PRNGKey(args.key)
    frames = []
    for batch in tqdm(traj.batch(args.batch)):
        root_key, key = jax.random.split(root_key, 2)
        pose = jax.tree_util.tree_map(jnp.array, batch)
        keys = jnp.array(jax.random.split(key, batch.x.shape[0]))

        frames.append(np.asarray(dart.render(state, pose)))

    out = {"rad": np.concatenate(frames, axis=0)}
    savemat(args.path + ".mat", out)
