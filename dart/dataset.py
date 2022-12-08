"""Datasets."""

from jax import numpy as jnp
from jax import vmap
import numpy as np
from tensorflow.data import Dataset
from scipy.io import loadmat

from .fields import GroundTruth
from .sample import VirtualRadar


def _load_arrays(file: str) -> any:
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


def trajectory(traj: str, sensor: VirtualRadar) -> Dataset:
    """Generate trajectory dataset."""
    traj = _load_arrays(traj)
    pose = vmap(sensor.make_pose)(
        traj["velocity"], traj["position"], traj["orientation"])
    return Dataset.from_tensor_slices(pose)


def dart(traj: str, images: str, sensor: VirtualRadar) -> Dataset:
    """Dataset with trajectory and images."""
    traj = _load_arrays(traj)
    images = _load_arrays(images)
    pose = vmap(sensor.make_pose)(
        traj["velocity"], traj["position"], traj["orientation"])
    return Dataset.from_tensor_slices((pose, images["y"]))
