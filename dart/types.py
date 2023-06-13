"""Common DART types."""

from argparse import ArgumentParser, _ArgumentGroup, Namespace

import numpy as np
import tensorflow as tf

from jaxtyping import Array, Float32, UInt8, Int32, UInt32, PyTree
from jaxtyping import Float16 as RadarFloat

from beartype.typing import Union, Callable, NamedTuple
from jax.random import PRNGKeyArray

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
    [Float32[Array, "3"]],
    tuple[Float32[Array, ""], Float32[Array, ""], Float32[Array, ""]]]

#: Antenna gain pattern
GainPattern = Callable[
    [Float32[Array, "k"], Float32[Array, "k"]], Float32[Array, "k Na"]]

#: Hyperparameters schedule
HyperparameterSchedule = Callable[[int, int], PyTree]


class CameraPose(NamedTuple):
    """Camera pose parameters for simple ray rendering.

    x: sensor location in global coordinates.
    A: 3D rotation matrix for sensor pose; should transform sensor-space to
        world-space.
    """

    x: Float32[Array, "3"]
    A: Float32[Array, "3 3"]


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

    v: Float32[Array, "3"]
    p: Float32[Array, "3"]
    q: Float32[Array, "3"]
    s: Float32[Array, ""]
    x: Float32[Array, "3"]
    A: Float32[Array, "3 3"]
    i: Int32[Array, ""]


class TrainingColumn(NamedTuple):
    """Single column for training.

    For 256 range bins and 256 angular bins, this takes::

        96 + 256 / 8 + 4 + 4 = 136 bytes.

    Attributes
    ----------
    pose: pose for each column (96 bytes).
    valid: validity of each angular bin; bit-packed bool array (n / 8 bytes).
    weight: velocity-corrected weight of each bin (4 bytes).
    doppler: doppler value for this column (4 bytes).
    """

    pose: RadarPose
    valid: UInt8[Array, "n8"]
    weight: Float32[Array, ""]
    doppler: RadarFloat[Array, ""]


#: Image data
RangeDopplerData = tuple[
    RadarPose, RadarFloat[Union[Array, np.ndarray], "Ni Nr Nd Na"]]

#: Doppler column data
DopplerColumnData = tuple[
    TrainingColumn, RadarFloat[Union[Array, np.ndarray], "Nc Nr Na"]]


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
