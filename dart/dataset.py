"""Datasets."""

import jax
from jax import numpy as jnp
import numpy as np

from scipy.io import loadmat
import h5py

from jaxtyping import Integer, Array, PyTree, Float
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
    return GroundTruth.from_occupancy(
        jnp.array(data['grid']), data["lower"], data["upper"])


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
    path: str, norm: float = 1e4, sensor: Optional[VirtualRadar] = None,
    threshold: float = 0.0
) -> types.RangeDopplerData:
    """Load image-trajectory data."""
    src = load_arrays(path)
    idxs = jnp.arange(src["vel"].shape[0])
    pose = jax.vmap(make_pose)(src["vel"], src["pos"], src["rot"], idxs)
    image = jnp.where(src["rad"] > threshold, src["rad"], threshold) / norm

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
            weight = sensor.n / pose.s
            return types.TrainingColumn(
                pose=pose, weight=weight, doppler=doppler)
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


def __doppler_decimation(
    rda: Float[types.ArrayLike, "Ni Nr Nd Na"], factor: int
) -> Float[types.ArrayLike, "Ni Nr Nd Na"]:
    """Apply doppler averaging decimation."""
    Ni, Nr, Nd, Na = rda.shape
    assert Nd % factor == 0
    decimated = jnp.mean(
        rda.reshape(Ni, Nr, Nd // factor, factor, Na), axis=3, keepdims=True)
    return jnp.tile(decimated, (1, 1, 1, factor, 1)).reshape(rda.shape)


def doppler_columns(
    sensor: VirtualRadar, path: str = "data/cup.mat", norm: float = 1e4,
    pval: float = 0., iid_val: bool = False, min_speed: float = 0.1,
    repeat: int = 0, threshold: float = 0.0, doppler_decimation: int = 0,
    key: types.PRNGSeed = 42
) -> tuple[types.Dataset, Optional[types.Dataset], dict[str, PyTree]]:
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
    threshold: Mask out values less than the provided threshold (set to 0).
    doppler_decimation: Simulate a lower doppler resolution by setting each
        block of consecutive doppler columns to their average.
    key: Random key to shuffle dataset frames. Does not shuffle columns.

    Returns
    -------
    train: Train dataset.
    val: Val dataset.
    validx: Indices of original images corresponding to the validation set.
    """
    pose, range_doppler = __raw_image_traj(
        path, norm=norm, sensor=sensor, threshold=threshold)

    if doppler_decimation > 0:
        range_doppler = __doppler_decimation(range_doppler, doppler_decimation)

    idx = jnp.arange(range_doppler.shape[0], dtype=jnp.int32)
    valid_speed = (pose.s > min_speed) & (pose.s < sensor.d[-1])

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

    return train, val, {"val": idx[-nval:]}
