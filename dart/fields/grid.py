"""Basic Plenoxels-inspired Grid."""

from jax import numpy as jnp
import haiku as hk
from jaxtyping import Float32, Array
from beartype.typing import Union, Optional, Callable

from dart import types
from ._spatial import interpolate


class SimpleGrid(hk.Module):
    """Simple reflectance/log-transmittance grid.

    Parameters
    ----------
    size: Grid size (x, y, z) dimensions.
    lower: Lower corner of the grid.
    resolution: Resolution in units per grid cell. Can have the same resolution
        for each axis or different resolutions.
    """

    _description = "voxel grid with trilinear interpolation"

    def __init__(
        self, size: tuple[int, int, int], lower: Float32[Array, "3"],
        resolution: Union[Float32[Array, "3"], Float32[Array, ""]],
    ) -> None:
        super().__init__()
        self.lower = lower
        self.resolution = resolution
        self.size = size

    def __call__(
        self, x: Float32[Array, "3"], dx: Optional[Float32[Array, "3"]] = None
    ) -> tuple[Float32[Array, ""], Float32[Array, ""]]:
        """Index into learned reflectance map."""
        grid = hk.get_parameter("grid", (*self.size, 2), init=jnp.zeros)
        index = (x - self.lower) * self.resolution
        valid = jnp.all((0 <= index) & (index <= jnp.array(self.size) - 1))
        sigma, alpha = jnp.where(
            valid, interpolate(index, grid), jnp.zeros((2,)))
        return sigma, alpha

    @classmethod
    def from_config(
        cls, size: list[int] = [512, 512, 256],
        lower: list[float] = [-4, -4, -1],
        resolution: Union[list[float], float] = [64.0, 64.0, 64.0]
    ) -> Callable[[], "SimpleGrid"]:
        """Create simple grid haiku closure from config items."""
        def closure():
            return cls(
                size=tuple(size), lower=jnp.array(lower),
                resolution=jnp.array(resolution))
        return closure

    @staticmethod
    def to_parser(p: types.ParserLike) -> None:
        """Create grid command line arguments."""
        p.add_argument(
            "--size", default=[256, 256, 128], nargs='+', type=int,
            help="Grid size (x, y, z)")
        p.add_argument(
            "--lower", default=[-4, -4, -1], nargs='+', type=float,
            help="Grid lower coordinate (x, y, z)")
        p.add_argument(
            "--resolution", default=[32.0, 32.0, 32.0], nargs='+', type=float,
            help="Grid resolution in grid cells per meter (x, y, z).")

    @classmethod
    def args_to_config(cls, args: types.Namespace) -> dict:
        """Create configuration dictionary."""
        assert len(args.size) == 3
        assert len(args.lower) == 3
        assert len(args.resolution) == 3
        return {
            "field_name": cls.__name__,
            "field": {
                "size": args.size, "lower": args.lower,
                "resolution": args.resolution
            }
        }
