"""Generate a simulated dataset from a ground truth reflectance grid."""

import json
import os
from tqdm import tqdm
import h5py
from functools import partial

import numpy as np
from jax import numpy as jnp
import jax

from dart import VirtualRadar, dataset, fields


def _parse(p):
    p.add_argument("-p", "--path", help="Path to data directory.")
    p.add_argument(
        "-r", "--key", default=42, type=int, help="Random seed.")
    p.add_argument("-o", "--out", default=None, help="Save path.")
    p.add_argument("-b", "--batch", default=16, type=int, help="Batch size")
    return p


def _main(args):

    if args.out is None:
        args.out = os.path.join(args.path, "simulated.h5")

    with open(os.path.join(args.path, "sensor.json")) as f:
        cfg = json.load(f)
    sensor = VirtualRadar.from_config(**cfg)

    gt_data = np.load(os.path.join(args.path, "map.npz"))
    gt = fields.GroundTruth.from_occupancy(
        jnp.array(gt_data['grid']), gt_data['lower'], gt_data['upper'])

    traj = dataset.trajectory(os.path.join(args.path, "data.h5"))

    render = partial(sensor.render, sigma=gt)
    render = jax.jit(jax.vmap(render))

    root_key = jax.random.PRNGKey(args.key)
    frames = []
    for batch in tqdm(traj.batch(args.batch)):
        root_key, key = jax.random.split(root_key, 2)
        pose = jax.tree_util.tree_map(jnp.array, batch)
        keys = jnp.array(jax.random.split(key, batch.x.shape[0]))

        frames.append(np.asarray(render(pose=pose, key=keys)))

    out = dataset.load_arrays(
        os.path.join(args.path, "data.h5"), keys=["pos", "vel"])
    out["rad"] = np.concatenate(frames, axis=0)

    with h5py.File(args.out, 'w') as hf:
        for k, v in out.items():
            hf.create_dataset(k, data=v)
