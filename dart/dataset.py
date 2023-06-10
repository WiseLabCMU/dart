"""Datasets."""

import jax
from jax import numpy as jnp
import numpy as np

from scipy.io import loadmat
import h5py

from jaxtyping import Integer, Array
from beartype.typing import Any, Optional

from .fields import GroundTruth
from .pose import make_pose
from .sensor import VirtualRadar
from . import types, utils


def _load_h5(
    file: str, keys: Optional[list[str]] = None, transpose: bool = False
) -> Any:
    """Load h5 file while handling matlab oddities."""
    f = h5py.File(file, 'r')
    key_res = f.keys() if keys is None else keys
    if transpose:
        return {k: np.array(f.get(k)).T for k in key_res}
    else:
        return {k: np.array(f.get(k)) for k in key_res}


def load_arrays(file: str, keys: Optional[list[str]] = None) -> Any:
    """General load function."""
    if file.endswith(".npz"):
        return np.load(file)
    elif file.endswith(".h5"):
        return _load_h5(file, keys)
    elif file.endswith(".mat"):
        try:
            return loadmat(file)
        except NotImplementedError:
            return _load_h5(file, keys, transpose=True)
        except ValueError:
            return _load_h5(file, keys)
    else:
        raise TypeError(
            "Unknown file type: {} (expected .npz, .h5, or .mat)".format(file))


def gt_map(file: str) -> GroundTruth:
    """Load ground truth reflectance map."""
    data = load_arrays(file)
    x, y, z = data['x'], data['y'], data['z']
    lower = jnp.array([np.min(x), np.min(y), np.min(z)])
    upper = jnp.array([np.max(x), np.max(y), np.max(z)])
    return GroundTruth.from_occupancy(
        jnp.array(data['v'], dtype=float), lower, upper, alpha=-100)


def trajectory(
    traj: str, subset: Optional[Integer[Array, "nval"]] = None
) -> types.Dataset:
    """Generate trajectory dataset."""
    data = load_arrays(traj)
    idxs = jnp.arange(data["vel"].shape[0])
    pose = jax.vmap(make_pose)(data["vel"], data["pos"], data["rot"], idxs)
    if subset is not None:
        pose = jax.tree_util.tree_map(lambda x: x[subset], pose)
    return types.Dataset.from_tensor_slices(pose)


def __raw_image_traj(
    path: str, norm: float = 1e4, sensor: Optional[VirtualRadar] = None
) -> types.RangeDopplerData:
    """Load image-trajectory data."""
    src = load_arrays(path)
    idxs = jnp.arange(src["vel"].shape[0])
    pose = jax.vmap(make_pose)(src["vel"], src["pos"], src["rot"], idxs)
    image = src["rad"] / norm
    if sensor is not None:
        if image.shape[1] > len(sensor.r):
            image = image[:, :len(sensor.r)]
        if image.shape[2] > len(sensor.d):
            crop = int((image.shape[2] - len(sensor.d)) / 2)
            image = image[:, :, crop:-crop]
        # Copy to garbage-collect the initial (larger) array
        image = np.copy(image)

    if len(image.shape) < 4:
        image = image.reshape(*image.shape, 1)

    return pose, image


def image_traj(
    path: str, norm: float = 1e4,
    subset: Optional[Integer[Array, "nval"]] = None
) -> types.Dataset:
    """Dataset with trajectory and images."""
    data = __raw_image_traj(path, norm=norm, sensor=None)
    if subset is not None:
        data = jax.tree_util.tree_map(lambda x: x[subset], (data))
    return types.Dataset.from_tensor_slices(data)


def __make_dataset(
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
    columns = jax.jit(jax.vmap(process_image))(poses)

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
    sensor: VirtualRadar, path: str = "data/cup.mat", norm: float = 1e4,
    pval: float = 0., iid_val: bool = False, min_speed: float = 0.1,
    repeat: int = 0, key: types.PRNGSeed = 42
) -> tuple[types.Dataset, Optional[types.Dataset], Integer[Array, "nval"]]:
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
    train: Train dataset.
    val: Val dataset.
    validx: Indices of original images corresponding to the validation set.
    """
    pose, range_doppler = __raw_image_traj(path, norm=norm, sensor=sensor)
    idx = np.arange(range_doppler.shape[0], dtype=np.int32)
    valid_speed = pose.s > min_speed

    print("Loaded dataset: {} valid frames (speed > {}) / {}".format(
        jnp.sum(valid_speed), min_speed, range_doppler.shape[0]))
    data = jax.tree_util.tree_map(
        lambda x: x[valid_speed], (pose, range_doppler, idx))

    if iid_val:
        data = utils.shuffle(data, key=key)

    pose, image, idx = data
    nval = 0 if pval <= 0 else int(utils.get_size(data) * pval)
    train, val = utils.split((pose, image), nval=nval)

    if not iid_val:
        train = utils.shuffle(train, key=key)

    train = __make_dataset(sensor, train)
    print("Train split : {} images --> {} valid columns".format(
        len(idx) - int(nval), train[1].shape))
    train = types.Dataset.from_tensor_slices(train)
    if val is not None:
        val = __make_dataset(sensor, val)
        print("Test split  : {} images --> {} valid columns".format(
            nval, val[1].shape))
        val = types.Dataset.from_tensor_slices(val)
    if repeat > 0:
        train = train.repeat(repeat)

    return train, val, idx[-nval:]
