"""Doppler column training data format."""

from jax import numpy as jnp

from jaxtyping import Float32, UInt8, Array
from beartype.typing import NamedTuple

from .pose import RadarPose
from .sensor import VirtualRadar


class TrainingColumn(NamedTuple):
    """Single column for training.

    For 256 range bins and 256 angular bins, this takes::

        96 + 256 / 8 + 4 + 4 + 256 * 4 = 1160 bytes.

    Attributes
    ----------
    pose: pose for each column (96 bytes).
    valid: validity of each angular bin; bit-packed bool array (n / 8 bytes).
    weight: number of valid bins (4 bytes).
    doppler: doppler value for this column (4 bytes).
    data: range-doppler data (n * 4 bytes).
    """

    pose: RadarPose
    valid: UInt8[Array, "n8"]
    weight: Float32[Array, ""]
    doppler: Float32[Array, ""]
    data: Float32[Array, "nr"]


def make_column(
    data: Float32[Array, "nr"], doppler: Float32[Array, ""], pose: RadarPose,
    sensor: VirtualRadar
) -> TrainingColumn:
    """Create column for training.

    Parameters
    ----------
    data: data in this column.
    d: doppler value.
    pose: sensor pose.
    sensor: sensor parameters.

    Returns
    -------
    Training point with per-computed valid bins.
    """
    psi = jnp.arange(sensor.n) * sensor.bin_width
    valid = sensor.valid_mask(doppler, psi, pose)
    packed = jnp.packbits(valid)
    weight = jnp.sum(valid).astype(jnp.float32)
    return TrainingColumn(
        pose=pose, valid=packed, weight=weight, doppler=doppler, data=data)
