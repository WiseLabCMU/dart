"""Ground truth field for simulations."""

from jax import numpy as jnp
from jaxtyping import Float32, Array
from beartype.typing import Union, Optional

from ._spatial import interpolate


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
        self, x: Float32[Array, "3"], **kwargs
    ) -> tuple[Float32[Array, ""], Float32[Array, ""]]:
        """Index into reflectance map."""
        index = (x - self.lower) * self.resolution
        valid = jnp.all(
            (0 <= index) & (index <= jnp.array(self.grid.shape[:-1]) - 1))
        sigma, alpha = jnp.where(
            valid, interpolate(index, self.grid), jnp.zeros((2,)))
        return sigma, alpha

    @classmethod
    def from_occupancy(
        cls, occupancy: Float32[Array, "Nx Ny Nz"],
        lower: Float32[Array, "3"], upper: Float32[Array, "3"],
        alpha: float = -100
    ) -> "GroundTruth":
        """Create ground truth from 0-1 occupancy grid.

        Parameters
        ----------
        occupancy: source occupancy grid (i.e. from 3D model)
        lower: coordinates of lower bound of the grid
        upper: coordinates of upper bound of the grid
        alpha: multiplier to apply to alpha of occupied cells
        """
        resolution = jnp.array(occupancy.shape) / (upper - lower)
        grid = jnp.stack([occupancy, alpha * occupancy], axis=-1)
        return cls(grid, lower=lower, resolution=resolution)
