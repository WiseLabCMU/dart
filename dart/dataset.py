"""Datasets."""

from beartype.typing import Any
from functools import partial

import jax
from jax import numpy as jnp
import numpy as np
from tensorflow.data import Dataset
from scipy.io import loadmat

from .fields import GroundTruth
from .pose import make_pose
from .sensor import VirtualRadar


def _load_arrays(file: str) -> Any:
    if file.endswith(".npz"):
        return np.load(file)
    elif file.endswith(".mat"):
        return loadmat(file)
    else:
        raise TypeError(
            "Unknown file type: {} (expected .npz or .mat)".format(file))


def gt_map(file: str) -> GroundTruth:
    """Load ground truth reflectance map."""
    data = _load_arrays(file)
    x = data['x']
    y = data['y']
    z = data['z']

    lower = jnp.array([np.min(x), np.min(y), np.min(z)])
    upper = jnp.array([np.max(x), np.max(y), np.max(z)])
    resolution = jnp.array(data['v'].shape) / (upper - lower)

    return GroundTruth(
        jnp.array(data['v'], dtype=float), lower=lower, resolution=resolution)


def trajectory(traj: str) -> Dataset:
    """Generate trajectory dataset."""
    traj = _load_arrays(traj)
    pose = jax.vmap(make_pose)(
        traj["velocity"], traj["position"], traj["orientation"])
    return Dataset.from_tensor_slices(pose)


def image_traj(traj: str, images: str) -> Dataset:
    """Dataset with trajectory and images."""
    traj = _load_arrays(traj)
    images = _load_arrays(images)
    pose = jax.vmap(make_pose)(
        traj["velocity"], traj["position"], traj["orientation"])
    return Dataset.from_tensor_slices((pose, images["y"]))


def dart(
        traj: str, images: str, sensor: VirtualRadar, pre_shuffle: bool = True
) -> Dataset:
    """Dataset with columns, poses, and other parameters."""
    traj = _load_arrays(traj)
    images = _load_arrays(images)['y']
    poses = jax.vmap(make_pose)(
        traj["velocity"], traj["position"], traj["orientation"])

    if pre_shuffle:
        indices = np.arange(images.shape[0])
        np.random.shuffle(indices)
        images = jax.tree_util.tree_map(lambda x: x[indices], images)
        poses = jax.tree_util.tree_map(lambda x: x[indices], poses)

    def process_image(img, pose):
        return jax.vmap(partial(sensor.make_column, pose=pose))(
            data=img.T, doppler=sensor.d)

    columns = jax.vmap(process_image)(images, poses)

    columns_flat = jax.tree_util.tree_map(
        lambda x: x.reshape(-1, *x.shape[2:]), columns)
    not_empty = columns_flat.weight > 0

    columns_valid = jax.tree_util.tree_map(
        lambda x: x[not_empty], columns_flat)
    return Dataset.from_tensor_slices(columns_valid)
