"""Reflectance field functions."""

from jaxtyping import Float32, Array
from beartype.typing import Union

from jax import numpy as jnp
import haiku as hk

from .spatial import interpolate


class GroundTruth:
    """Ground truth reflectance map.

    Parameters
    ----------
    grid: Reflectance grid; is trilinearly interpolated.
    lower: Lower corner of the grid.
    resolution: Resolution in units per grid cell. Can have the same resolution
        for each axis or different resolutions.
    """

    def __init__(
        self, grid: Float32[Array, "nx ny nz"], lower: Float32[Array, "3"],
        resolution: Union[Float32[Array, "3"], Float32[Array, ""]]
    ) -> None:
        self.lower = lower
        self.resolution = resolution
        self.grid = grid

    def __call__(self, x: Float32[Array, "3"]) -> Float32[Array, ""]:
        """Index into reflectance map."""
        index = (x - self.lower) * self.resolution
        valid = jnp.all(
            (0 <= index) & (index <= jnp.array(self.grid.shape) - 1))
        return jnp.where(
            valid,
            interpolate(index, self.grid.reshape(*self.grid.shape, 1))[0],
            jnp.zeros(()))


class SimpleGrid(hk.Module):
    """Simple reflectance grid."""

    def __init__(
        self, size: tuple[int, int, int], lower: Float32[Array, "3"],
        resolution: Union[Float32[Array, "3"], Float32[Array, ""]],
    ) -> None:
        super().__init__()
        self.lower = lower
        self.resolution = resolution
        self.size = size

    def __call__(self, x: Float32[Array, "3"]) -> Float32[Array, ""]:
        """Index into learned reflectance map."""
        grid = hk.get_parameter("grid", self.size, init=jnp.zeros)
        index = (x - self.lower) * self.resolution
        valid = jnp.all(
            (0 <= index) & (index <= jnp.array(self.size) - 1))
        return jnp.where(
            valid,
            interpolate(index, grid.reshape(*grid.shape, 1))[0],
            jnp.zeros(()))


class NGPHashTable:
    """Single hash table for NGP field.

    References
    ----------
    [1] Muller et al, "Instant Neural Graphics Primitives with a
        Multiresolution Hash Encoding," 2022.
    """

    def __init__(self, size):
        self.size = size

    def hash(self, x):
        """Apply hash function specified by NGP (Eq. 4 [1])."""
        pi2 = 2654435761
        pi3 = 805459861

        return (x[0] + x[1] * pi2 + x[2] * pi3) % self.size
