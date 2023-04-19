"""Datasets."""

import jax
from jax import numpy as jnp
import numpy as np

from tensorflow.data import Dataset
from scipy.io import loadmat
import h5py

from beartype.typing import Any, Optional

from .fields import GroundTruth
from .pose import make_pose
from .sensor import VirtualRadar
from . import types, utils


def load_arrays(file: str) -> Any:
    """General load function."""
    if file.endswith(".npz"):
        return np.load(file)
    elif file.endswith(".mat"):
        try:
            return loadmat(file)
        except NotImplementedError:
            f = h5py.File(file, 'r')
            return {k: np.array(f.get(k)).T for k in f.keys()}
    else:
        raise TypeError(
            "Unknown file type: {} (expected .npz or .mat)".format(file))


def gt_map(file: str) -> GroundTruth:
    """Load ground truth reflectance map."""
    data = load_arrays(file)
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
    data = load_arrays(traj)
    pose = jax.vmap(make_pose)(data["vel"], data["pos"], data["rot"])
    return Dataset.from_tensor_slices(pose)


def image_traj(path: str, clip: float = 99.9, norm: float = 0.05) -> Dataset:
    """Dataset with trajectory and images."""
    data = load_arrays(path)
    poses = jax.vmap(make_pose)(data["vel"], data["pos"], data["rot"])

    if clip > 0:
        images = data["rad"] / np.percentile(data["rad"], clip) * norm
    else:
        images = data["rad"]

    return Dataset.from_tensor_slices((poses, images))


def _make_dataset(
    sensor: VirtualRadar, data: types.RangeDopplerData
) -> types.DopplerColumnData:
    """Split poses/images into columns."""
    def process_image(pose):
        def make_column(doppler):
            valid = sensor.valid_mask(doppler, pose)
            packed = jnp.packbits(valid)
            weight = jnp.sum(valid).astype(jnp.float32) / pose.s
            return types.TrainingColumn(
                pose=pose, valid=packed, weight=weight, doppler=doppler)
        return jax.vmap(make_column)(sensor.d)

    poses, images = data
    columns = jax.vmap(process_image)(poses)
    images_col = jnp.swapaxes(images, 1, 2)
    dataset = (columns, images_col)

    # Flatten (index, doppler) order
    flattened = jax.tree_util.tree_map(
        lambda x: x.reshape(-1, *x.shape[2:]), dataset)

    # Remove invalid doppler columns
    not_empty = flattened[0].weight > 0
    dataset_valid = jax.tree_util.tree_map(lambda x: x[not_empty], flattened)

    return dataset_valid


def doppler_columns(
    sensor: VirtualRadar, path: str = "data/cup.mat", norm: float = 0.05,
    pval: float = 0., iid_val: bool = False, min_speed: float = 0.1,
    repeat: int = 0, key: types.PRNGSeed = 42
) -> tuple[Dataset, Optional[Dataset]]:
    """Load dataset trajectory and images.

    The dataset is ordered by::

        (image/pose index, doppler)

    With the image/pose shuffled. If the sensor has fewer range bins than
    are provided in the dataset, only the closest are used, and further
    bins are cropped out and removed.

    Parameters
    ----------
    sensor: Sensor profile for this dataset.
    path: Path to file containing data.
    norm: Normalization factor.
    pval: Proportion of dataset to hold as a validation set. If `pval=0`,
        no validation datset is returned.
    iid_val: If True, then shuffles the dataset before training so that the
        validation split is drawn randomly from the dataset instead of just
        from the end.
    min_speed: Minimum speed for usable samples. Images with lower
        velocities are rejected.
    repeat: Repeat dataset within each epoch to reduce overhead.
    key: Random key to shuffle dataset frames. Does not shuffle columns.

    Returns
    -------
    (train, val) datasets.
    """
    src = load_arrays(path)

    data = (
        jax.vmap(make_pose)(src["vel"], src["pos"], src["rot"]),
        src["rad"][:, :len(sensor.r)] / norm)

    valid_speed = data[0].s > min_speed

    print("Loaded dataset: {} valid frames (speed > {}) / {}".format(
        jnp.sum(valid_speed), min_speed, data[1].shape[0]))
    data = jax.tree_util.tree_map(lambda x: x[valid_speed], data)

    if iid_val:
        data = utils.shuffle(data, key=key)

    nval = 0 if pval <= 0 else int(utils.get_size(data) * pval)
    if nval > 0:
        train = jax.tree_util.tree_map(lambda x: x[:-nval], data)
        val = jax.tree_util.tree_map(lambda x: x[-nval:], data)
        val = _make_dataset(sensor, val)
        print("Test split  : {} images --> {} valid columns".format(
            nval, val[1].shape))
        valset = Dataset.from_tensor_slices(val)
    else:
        train = data
        valset = None

    if not iid_val:
        train = utils.shuffle(train, key=key)

    train = _make_dataset(sensor, train)
    trainset = Dataset.from_tensor_slices(train)
    print("Train split : {} images --> {} valid columns".format(
        data[1].shape[0] - int(nval), train[1].shape))

    if repeat > 0:
        trainset = trainset.repeat(repeat)
    return trainset, valset
