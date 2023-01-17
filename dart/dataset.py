"""Datasets."""

import jax
from jax import numpy as jnp
import numpy as np
from tensorflow.data import Dataset
from scipy.io import loadmat

from beartype.typing import Any

from .fields import GroundTruth
from .pose import make_pose


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
    grid = jnp.stack([occupancy, 1 - occupancy], axis=-1)

    return GroundTruth(
        grid, lower=lower, resolution=resolution)


def trajectory(traj: str) -> Dataset:
    """Generate trajectory dataset."""
    traj = _load_arrays(traj)
    pose = jax.vmap(make_pose)(traj["vel"], traj["pos"], traj["rot"])
    return Dataset.from_tensor_slices(pose)


def image_traj(path: str, clip: float = 99.9) -> Dataset:
    """Dataset with trajectory and images."""
    data = _load_arrays(path)
    poses = jax.vmap(make_pose)(data["vel"], data["pos"], data["rot"])
    images = data["rad"] / np.percentile(data["rad"], clip)

    return Dataset.from_tensor_slices((poses, images))
