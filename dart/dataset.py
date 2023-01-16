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

    occupancy = jnp.array(data['v'], dtype=float)
    grid = jnp.stack([occupancy, occupancy], axis=-1)

    return GroundTruth(
        grid, lower=lower, resolution=resolution)


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


def image_traj2(path: str) -> Dataset:
    """Dataset with trajectory and images."""
    data = _load_arrays(path)
    poses = jax.vmap(make_pose)(data["vel"], data["pos"], data["rot"])

    return Dataset.from_tensor_slices((poses, data["rad"]))


def dart(
        traj: str, images: str, sensor: VirtualRadar, pre_shuffle: bool = True
) -> Dataset:
    """Dataset with columns, poses, and other parameters.

    The dataset is ordered by::

        (image/pose index, doppler)

    With only the image/pose "pre-shuffled."
    """
    traj = _load_arrays(traj)
    images = _load_arrays(images)['y']
    poses = jax.vmap(make_pose)(
        traj["velocity"], traj["position"], traj["orientation"])

    if pre_shuffle:
        indices = np.arange(images.shape[0])
        np.random.shuffle(indices)
        images = jax.tree_util.tree_map(lambda x: x[indices], images)
        poses = jax.tree_util.tree_map(lambda x: x[indices], poses)

    def process_image(pose):
        return jax.vmap(
            partial(sensor.make_column, pose=pose))(doppler=sensor.d)

    columns = jax.vmap(process_image)(poses)
    images_col = jnp.swapaxes(images, 1, 2)
    dataset = (columns, images_col)

    dataset_flat = jax.tree_util.tree_map(
        lambda x: x.reshape(-1, *x.shape[2:]), dataset)
    not_empty = dataset_flat[0].weight > 0

    dataset_valid = jax.tree_util.tree_map(
        lambda x: x[not_empty], dataset_flat)
    return Dataset.from_tensor_slices(dataset_valid)


def dart2(
        path: str, sensor: VirtualRadar, pre_shuffle: bool = True) -> Dataset:
    """Real dataset with all in one.

    The dataset is ordered by::

        (image/pose index, doppler)

    With only the image/pose "pre-shuffled."
    """
    data = _load_arrays(path)
    poses = jax.vmap(make_pose)(data["vel"], data["pos"], data["rot"])

    if pre_shuffle:
        indices = np.arange(data["rad"].shape[0])
        np.random.shuffle(indices)
        images = data["rad"]
        images = jax.tree_util.tree_map(lambda x: x[indices], images)
        images = jnp.sqrt(images) / 10000
        poses = jax.tree_util.tree_map(lambda x: x[indices], poses)

    def process_image(pose):
        return jax.vmap(
            partial(sensor.make_column, pose=pose))(doppler=sensor.d)

    columns = jax.vmap(process_image)(poses)
    images_col = jnp.swapaxes(images, 1, 2)
    dataset = (columns, images_col)

    dataset_flat = jax.tree_util.tree_map(
        lambda x: x.reshape(-1, *x.shape[2:]), dataset)
    not_empty = dataset_flat[0].weight > 0

    dataset_valid = jax.tree_util.tree_map(
        lambda x: x[not_empty], dataset_flat)

    print(dataset_valid[1].shape)

    return Dataset.from_tensor_slices(dataset_valid).repeat(100)
