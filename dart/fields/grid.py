"""Basic Plenoxels-inspired Grid."""

from jax import numpy as jnp
import haiku as hk
import math
from jaxtyping import Float32, Array
from beartype.typing import Optional, Callable

from dart import types
from ._spatial import interpolate


class VoxelGrid(hk.Module):
    """Reflectance/log-transmittance voxel grid.

    Parameters
    ----------
    size: Grid size (x, y, z) dimensions.
    lower: Lower corner of the grid.
    upper: Upper corner of the grid.
    size: Size of the grid.
    resolution: Resolution in units per grid cell; must be precomputed and
        provided.
    """

    _description = "voxel grid with trilinear interpolation"

    def __init__(
        self, lower: Float32[Array, "3"], upper: Float32[Array, "3"],
        resolution: float, size: list[int], do_alpha: bool = False
    ) -> None:
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.resolution = resolution
        self.do_alpha = do_alpha
        self.size = size

    def __call__(
        self, x: Float32[Array, "3"], dx: Optional[Float32[Array, "3"]] = None,
        **kwargs
    ) -> tuple[Float32[Array, ""], Float32[Array, ""]]:
        """Index into learned reflectance map."""
        index = (x - self.lower) * self.resolution
        valid = jnp.all((0 <= index) & (index <= jnp.array(self.size) - 1))
        grid = jnp.array(hk.get_parameter(
            "grid", (*self.size, 2 if self.do_alpha else 1), init=jnp.zeros))

        if self.do_alpha:
            sigma, alpha = jnp.where(
                valid, interpolate(index, grid), jnp.zeros((2,)))
        else:
            sigma = jnp.where(
                valid, interpolate(index, grid), jnp.zeros((1,)))[0]
            alpha = jnp.array(0.0)

        return sigma, alpha

    @classmethod
    def from_config(
        cls, lower: list[float] = [-4.0, -4.0, -1.0], do_alpha: bool = False,
        upper: list[float] = [4.0, 4.0, 1.0], resolution: float = 25.0,
        size: list[int] = [100, 100, 100]
    ) -> Callable[[], "VoxelGrid"]:
        """Create simple grid haiku closure from config items."""
        def closure():
            return cls(
                upper=jnp.array(upper), lower=jnp.array(lower),
                resolution=resolution, do_alpha=do_alpha, size=size)
        return closure

    @staticmethod
    def to_parser(p: types.ParserLike) -> None:
        """Create grid command line arguments."""
        p.add_argument(
            "--lower", default=[-4.0, -4.0, -1.0], nargs='+', type=float,
            help="Lower coordinate (x, y, z).")
        p.add_argument(
            "--upper", default=[4.0, 4.0, 1.0], nargs='+', type=float,
            help="Upper coordinate (x, y, z).")
        p.add_argument(
            "--resolution", default=25.0, type=float,
            help="Grid resolution in grid cells per meter.")
        p.add_argument(
            "--do_alpha", default=False, action='store_true',
            help="Enable opacity/transmittance parameter.")

    @classmethod
    def args_to_config(cls, args: types.Namespace) -> dict:
        """Create configuration dictionary."""
        assert len(args.upper) == 3
        assert len(args.lower) == 3
        grid_size = [
            math.ceil((l - u) * args.resolution)
            for u, l in zip(args.lower, args.upper)]
        return {
            "field_name": cls.__name__,
            "field": {
                "lower": args.lower, "upper": args.upper,
                "resolution": args.resolution, "size": grid_size,
                "do_alpha": args.do_alpha
            }
        }
