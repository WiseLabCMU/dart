"""Ground truth field for simulations."""

from jaxtyping import Float32, Array
from beartype.typing import Union, Optional

from jax import numpy as jnp
from dart.spatial import interpolate


class GroundTruth:
    """Ground truth reflectance map.

    Parameters
    ----------
    grid: (reflectance, transmittance) grid; is trilinearly interpolated.
    lower: Lower corner of the grid.
    resolution: Resolution in units per grid cell. Can have the same resolution
        for each axis or different resolutions.
    """

    def __init__(
        self, grid: Float32[Array, "nx ny nz 2"], lower: Float32[Array, "3"],
        resolution: Union[Float32[Array, "3"], Float32[Array, ""]]
    ) -> None:
        self.lower = lower
        self.resolution = resolution
        self.grid = grid

    def __call__(
        self, x: Float32[Array, "3"], dx: Optional[Float32[Array, "3"]] = None
    ) -> tuple[Float32[Array, ""], Float32[Array, ""]]:
        """Index into reflectance map."""
        index = (x - self.lower) * self.resolution
        valid = jnp.all(
            (0 <= index) & (index <= jnp.array(self.grid.shape[:-1]) - 1))
        sigma, alpha = jnp.where(
            valid, interpolate(index, self.grid), jnp.zeros((2,)))
        return sigma, alpha
