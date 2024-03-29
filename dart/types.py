"""Common DART types."""

from argparse import ArgumentParser, _ArgumentGroup, Namespace

import numpy as np
import tensorflow as tf
import h5py

from jaxtyping import Array, Float32, Int32, UInt32, PyTree, Float, PRNGKeyArray
from jaxtyping import Float16 as RadarFloat
from beartype.typing import Union, Callable, NamedTuple


#: Argument parser or parser group
ParserLike = Union[ArgumentParser, _ArgumentGroup]

#: Parsed arguments
ParsedArgs = Namespace

#: Dataset needs to be manually "imported"
#: https://github.com/microsoft/pylance-release/issues/1066
Dataset = tf.data.Dataset

#: PRNGKey, according to JAX alias.
PRNGKey = Union[PRNGKeyArray, UInt32[Array, "... 2"]]

#: PRNGKey seed. Can be int (passed to jax.random.PRNGKey) or PRNGKey.
PRNGSeed = Union[int, PRNGKey]

#: Reflectance field
SigmaField = Callable[
    [Float32[Array, "3"], Float32[Array, "3"]],
    tuple[Float32[Array, ""], Float32[Array, ""]]]

#: Antenna gain pattern
GainPattern = Callable[
    [Float32[Array, "k"], Float32[Array, "k"]], Float32[Array, "k Na"]]

#: Hyperparameters schedule
HyperparameterSchedule = Callable[[int, int], PyTree]

#: Numpy or JAX
ArrayLike = Union[Array, np.ndarray]

#: Scalar or array
FloatLike = Union[Float[ArrayLike, ""], float]


class CameraPose(NamedTuple):
    """Camera pose parameters for simple ray rendering.

    x: sensor location in global coordinates.
    A: 3D rotation matrix for sensor pose; should transform sensor-space to
        world-space.
    """

    x: Float32[Array, "... 3"]
    A: Float32[Array, "... 3 3"]


class RadarPose(NamedTuple):
    """Radar pose parameters (24 x 4 = 96 bytes).

    Attributes
    ----------
    v: normalized velocity direction (``||v||_2=1``).
    p, q: orthonormal basis along with ``v``.
    s: speed (magnitude of un-normalized velocity).
    x: sensor location in global coordinates.
    A: 3D rotation matrix for sensor pose; should transform sensor-space to
        world-space.
    i: index of pose relative to the original order.
    """

    v: Float32[ArrayLike, "... 3"]
    p: Float32[ArrayLike, "... 3"]
    q: Float32[ArrayLike, "... 3"]
    s: Float32[ArrayLike, "..."]
    x: Float32[ArrayLike, "... 3"]
    A: Float32[ArrayLike, "... 3 3"]
    i: Int32[ArrayLike, "..."]

    @classmethod
    def from_h5file(cls, file) -> "RadarPose":
        """Load from file."""
        return cls(**{
            k: np.array(file[k]) for k in ['v', 'p', 'q', 's', 'x', 'A', 'i']})

    def to_h5file(self, file) -> None:
        """Save to file."""
        for key in ['v', 'p', 'q', 's', 'x', 'A', 'i']:
            file.create_dataset(key, data=getattr(self, key))


class TrainingColumn(NamedTuple):
    """Single column for training.

    For 256 range bins and 256 angular bins, this takes::

        96 + 4 + 2 = 102 bytes.

    Attributes
    ----------
    pose: pose for each column (96 bytes).
    weight: velocity-corrected weight of each bin (4 bytes).
    doppler: doppler value for this column (2 bytes).
    """

    pose: RadarPose
    weight: Float32[ArrayLike, "..."]
    doppler: RadarFloat[ArrayLike, "..."]


#: Image data
RangeDopplerData = tuple[RadarPose, RadarFloat[ArrayLike, "Ni Nr Nd Na"]]

#: Doppler column data
DopplerColumnData = tuple[TrainingColumn, RadarFloat[ArrayLike, "Nc Nr Na"]]


class ModelState(NamedTuple):
    """Model parameters and optimizer state.

    Attributes
    ----------
    params: main model parameters
    # delta: trajectory adjustment parameters
    opt_state: optimizer state for the main model
    """

    params: PyTree
    # delta: Optional[PyTree]
    opt_state: PyTree

    @staticmethod
    def get_params(x: Union[PyTree, "ModelState"]) -> PyTree:
        """Get params from the union type."""
        if isinstance(x, ModelState):
            return x.params
        else:
            return x


class Average(NamedTuple):
    """Loss averaging."""

    avg: float
    n: float
