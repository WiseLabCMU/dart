"""Spatial utilities."""

from jaxtyping import Float32, Array, jaxtyped
from beartype import beartype as typechecker
from beartype.typing import Union, Callable

from jax import numpy as jnp

# todo: move to another file
import haiku as hk
from beartype.typing import Tuple, Int


@jaxtyped
@typechecker
def interpolate(
    x: Float32[Array, "3"],
    grid: Union[
        Float32[Array, "nx ny nz d"],
        Callable[[Float32[Array, "8 3"]], Float32[Array, "d"]]
    ] = None
) -> Float32[Array, "d"]:
    """Trilinear 3D Interpolation (jit + vmap safe).

    Supports n-dimensional features; results in 8 grid calls/accesses.

    Parameters
    ----------
    x : Coordinates relative to the grid. Should have 0 <= x <= grid.shape.
    grid : ``(x, y, z) -> (d,)`` Grid accessor function; the function is
        responsible for its own vectorization. If passed as just an
        ``(nx, ny, nz, d)`` array, accesses naively.

    Returns
    -------
    Vector interpolated value with dimension ``d``.
    """
    # 8 corners; compute at runtime since it's almost certainly faster than
    # loading in a constant from memory
    mask = ((
        jnp.arange(8).reshape(-1, 1)
        & jnp.left_shift(1, jnp.arange(3)).reshape(1, -1)
    ) != 0).astype(int)
    # (0, 0, 0) and (1, 1, 1) cube bounds
    bounds = jnp.stack([jnp.floor(x), jnp.ceil(x)]).astype(int)
    # Corner values
    c = bounds[mask, [0, 1, 2]]
    if isinstance(grid, jnp.DeviceArray):
        values = grid[c[:, 0], c[:, 1], c[:, 2]].reshape(8, -1)
    else:
        values = grid(c).reshape(8, -1)
    # Distances from opposite corner (same coord -> make weight 1)
    dist = jnp.abs(bounds[1 - mask, [0, 1, 2]].astype(float) - x)
    dist += (dist == 0)
    # Opposite volume (https://en.wikipedia.org/wiki/Trilinear_interpolation)
    weights = jnp.prod(dist, axis=1).reshape(8, -1)
    return jnp.sum(values * weights / jnp.sum(weights), axis=0)


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

    def __call__(self, x: Float32[Array, "3"]):
        """Index into reflectance map."""
        index = (x - self.lower) / self.resolution
        valid = jnp.all(
            (0 <= index) & (index <= jnp.array(self.grid.shape) - 1))
        return jnp.where(valid, interpolate(index, self.grid), jnp.zeros(()))


class SimpleGrid(hk.Module):
    """Simple reflectance grid."""

    def __init__(
        self, size: Tuple[int, int, int], lower: Float32[Array, "3"],
        resolution: Union[Float32[Array, "3"], Float32[Array, ""]],
    ) -> None:
        self.lower = lower
        self.resolution = resolution
        self.size = size

    def __call__(self, x: Float32[Array, "3"]):
        """Index into learned reflectance map."""
        grid = self.get_parameter("grid", self.size, init=jnp.zeros)
        index = (x - self.lower) / self.resolution
        valid = jnp.all(
            (0 <= index) & (index <= jnp.array(self.size) - 1))
        return jnp.where(valid, interpolate(index, grid), jnp.zeros(()))
