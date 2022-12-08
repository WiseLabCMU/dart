"""Reflectance field functions."""

from jaxtyping import Float32, Array, jaxtyped
from beartype import beartype as typechecker
from beartype.typing import Union, Tuple

from jax import numpy as jnp
import haiku as hk

from .spatial import interpolate


@jaxtyped
@typechecker
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


@jaxtyped
@typechecker
class SimpleGrid(hk.Module):
    """Simple reflectance grid."""

    def __init__(
        self, size: Tuple[int, int, int], lower: Float32[Array, "3"],
        resolution: Union[Float32[Array, "3"], Float32[Array, ""]],
    ) -> None:
        self.lower = lower
        self.resolution = resolution
        self.size = size

    def __call__(self, x: Float32[Array, "3"]) -> Float32[Array, ""]:
        """Index into learned reflectance map."""
        grid = self.get_parameter("grid", self.size, init=jnp.zeros)
        index = (x - self.lower) / self.resolution
        valid = jnp.all(
            (0 <= index) & (index <= jnp.array(self.size) - 1))
        return jnp.where(valid, interpolate(index, grid), jnp.zeros(()))
