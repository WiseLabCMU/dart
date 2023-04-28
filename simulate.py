"""Generate simulated data."""

import json
from tqdm import tqdm
from argparse import ArgumentParser
from functools import partial

import numpy as np
from jax import numpy as jnp
import jax
from scipy.io import savemat

from dart import VirtualRadar, dataset


def _parse():

    p = ArgumentParser()
    p.add_argument(
        "-s", "--sensor", default="data/sim96.json",
        help="Sensor configuration.")
    p.add_argument(
        "-r", "--key", default=42, type=int, help="Random seed.")
    p.add_argument("-o", "--out", default="simulated", help="Save path.")
    p.add_argument(
        "-g", "--gt", default="data/map.mat", help="Ground truth reflectance.")
    p.add_argument(
        "-j", "--traj", default="data/traj.mat", help="Sensor trajectory.")
    p.add_argument("-b", "--batch", default=64, type=int, help="Batch size")
    return p


if __name__ == '__main__':

    args = _parse().parse_args()

    with open(args.sensor) as f:
        cfg = json.load(f)
    sensor = VirtualRadar.from_config(**cfg)

    gt = dataset.gt_map(args.gt)
    traj = dataset.trajectory(args.traj)

    render = partial(sensor.render, sigma=gt)
    render = jax.jit(jax.vmap(render))

    root_key = jax.random.PRNGKey(args.key)
    frames = []
    for batch in tqdm(traj.batch(args.batch)):
        root_key, key = jax.random.split(root_key, 2)
        pose = jax.tree_util.tree_map(jnp.array, batch)
        keys = jnp.array(jax.random.split(key, batch.x.shape[0]))

        frames.append(np.asarray(render(pose=pose, key=keys)))

    out = dataset.load_arrays(args.traj)
    out = {k: v for k, v in out.items() if not k.startswith("__")}
    out["rad"] = np.concatenate(frames, axis=0)

    savemat(args.out, out)
