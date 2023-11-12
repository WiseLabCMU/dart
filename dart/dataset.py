"""Datasets."""

import jax
from jax import numpy as jnp
import numpy as np

from scipy.io import loadmat
import h5py

from jaxtyping import Integer, PyTree, Float
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


def trajectory(
    traj: str, subset: Optional[Integer[types.ArrayLike, "nval"]] = None
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
    pose = jax.vmap(make_pose)(
        src["vel"], src["pos"], src["rot"], idxs)
    image = (np.maximum(src["rad"], threshold) / norm).astype(np.float16)

    if sensor is not None:
        if image.shape[1] > len(sensor.r):
            image = image[:, :len(sensor.r)]
        if image.shape[2] > len(sensor.d):
            crop = int((image.shape[2] - len(sensor.d)) / 2)
            image = image[:, :, crop:-crop]
        image = np.copy(image)

    if len(image.shape) < 4:
        image = image.reshape(*image.shape, 1)

    return pose, image


def image_traj(
    path: str, norm: float = 1e4,
    subset: Optional[Integer[types.ArrayLike, "nval"]] = None
) -> types.Dataset:
    """Dataset with trajectory and images."""
    data = __raw_image_traj(path, norm=norm, sensor=None)
    if subset is not None:
        data = jax.tree_util.tree_map(lambda x: x[subset], (data))
    return types.Dataset.from_tensor_slices(data)


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
    path: str, pval: float = 0., iid_val: bool = False,
    doppler_decimation: int = 0,
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
    path: Path to file containing data.
    pval: Proportion of dataset to hold as a validation set. If `pval=0`,
        no validation dataset is returned.
    iid_val: If True, then shuffles the dataset before training so that the
        validation split is drawn randomly from the dataset instead of just
        from the end.
    key: Random key to shuffle dataset frames. Does not shuffle columns.

    Returns
    -------
    train: Train dataset.
    val: Val dataset.
    meta: Metadata (exact split indices).
    """
    file = h5py.File(path)

    pose = types.RadarPose.from_h5file(file)
    rad = np.array(file["rad"], dtype=np.float16)
    weight = np.array(file["weight"], dtype=np.float32)
    doppler = np.array(file["doppler"], dtype=np.float32)
    idx = np.arange(rad.shape[0])

    meta = types.TrainingColumn(pose=pose, weight=weight, doppler=doppler)
    data = (meta, rad), idx

    print("Loaded dataset : {} valid columns".format(rad.shape))

    if iid_val:
        data = utils.shuffle(data, key=key)

    nval = 0 if pval <= 0 else int(rad.shape[0] * pval)
    (train, itrain), _val = utils.split(data, nval=nval)

    if not iid_val:
        train = utils.shuffle(train, key=key)

    print("Train split    : {} columns".format(train[1].shape))
    train = types.Dataset.from_tensor_slices(train)
    if _val is not None:
        val, ival = _val
        print("Test split     : {} columns".format(val[1].shape))
        val = types.Dataset.from_tensor_slices(val)
    else:
        val = None
        ival = np.zeros(0, dtype=bool)

    return train, val, {"train": itrain, "val": ival}
