"""Generate a simulated dataset from a ground truth reflectance grid."""

import json
import os
from tqdm import tqdm
import h5py
from functools import partial

import numpy as np
from jax import numpy as jnp
import jax

from dart import VirtualRadar, fields, pose, types


def _parse(p):
    p.add_argument("-p", "--path", help="Path to data directory.")
    p.add_argument(
        "-r", "--key", default=42, type=int, help="Random seed.")
    p.add_argument("-o", "--out", default=None, help="Save path.")
    p.add_argument("-b", "--batch", default=16, type=int, help="Batch size")
    return p


def _load_poses(path):
    data = h5py.File(path)
    vel = jnp.array(data["vel"])
    pos = jnp.array(data["pos"])
    rot = jnp.array(data["rot"])
    poses = jax.vmap(pose.make_pose)(vel, pos, rot, jnp.arange(vel.shape[0]))
    return types.Dataset.from_tensor_slices(poses)


def _main(args):
    if args.out is None:
        args.out = os.path.join(args.path, "simulated.h5")

    sensor = VirtualRadar.from_file(args.path)
    gt_data = np.load(os.path.join(args.path, "map.npz"))
    gt = fields.GroundTruth.from_occupancy(
        jnp.array(gt_data['grid']), gt_data['lower'], gt_data['upper'],
        alpha_scale=100.0)

    traj = _load_poses(os.path.join(args.path, "trajectory.h5"))

    render = partial(sensor.render, sigma=gt)
    render = jax.jit(jax.vmap(render))

    root_key = jax.random.PRNGKey(args.key)
    frames = []
    for batch in tqdm(traj.batch(args.batch)):
        root_key, key = jax.random.split(root_key, 2)
        keys = jnp.array(jax.random.split(key, batch.x.shape[0]))
        pose = jax.tree_util.tree_map(jnp.array, batch)
        frames.append(
            np.asarray(render(pose=pose, key=keys), dtype=np.float16))

    with h5py.File(args.out, 'w') as hf:
        hf.create_dataset("rad", data=np.concatenate(frames, axis=0))
