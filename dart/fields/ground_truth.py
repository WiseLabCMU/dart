"""Ground truth field for simulations."""

from jax import numpy as jnp

from jaxtyping import Float32, Array, Bool, Float
from beartype.typing import Optional

from ._spatial import interpolate
from dart import types


class GroundTruth:
    """Ground truth reflectance map.

    Parameters
    ----------
    grid: reflectance grid; is trilinearly interpolated.
    lower: Lower corner of the grid.
    resolution: Resolution in units per grid cell.
    """

    def __init__(
        self, grid: Float[Array, "nx ny nz"],
        lower: Float[types.ArrayLike, "3"],
        resolution: Float[types.ArrayLike, "3"]
    ) -> None:
        self.lower = lower
        self.resolution = resolution
        self.grid = grid

    def __call__(
        self, x: Float32[Array, "3"], dx: Optional[Float32[Array, "3"]] = None,
        **kwargs
    ) -> tuple[Float32[Array, ""], Float32[Array, ""]]:
        """Index into reflectance map."""
        index = (x - self.lower) * self.resolution
        valid = jnp.all(
            (0 <= index) & (index <= jnp.array(self.grid.shape) - 1))
        sigma = jnp.where(
            valid, interpolate(index, self.grid[..., None]), 0.0)[0]
        return sigma, jnp.array(0.0)

    @classmethod
    def from_occupancy(
        cls, occupancy: Bool[Array, "Nx Ny Nz"],
        lower: Float[types.ArrayLike, "3"], upper: Float[types.ArrayLike, "3"],
    ) -> "GroundTruth":
        """Create ground truth from 0-1 occupancy grid.

        Parameters
        ----------
        occupancy: source occupancy grid (i.e. from 3D model)
        lower: coordinates of lower bound of the grid
        upper: coordinates of upper bound of the grid
        """
        resolution = jnp.array(occupancy.shape) / (upper - lower)
        grid = jnp.array(occupancy).astype(jnp.float16)
        return cls(grid, lower=lower, resolution=resolution)
