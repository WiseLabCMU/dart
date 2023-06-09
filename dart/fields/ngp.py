"""Instant Neural Graphics Primitive (NGP) based fields."""

from jax import numpy as jnp
import jax
import haiku as hk

from jaxtyping import Float32, Integer, Array
from beartype.typing import Optional, Callable

from dart import types
from ._spatial import interpolate, spherical_harmonics


class NGP(hk.Module):
    """NGP field [1].

    Parameters
    ----------
    levels: Resolution of each hash table level. The length determines the
        number of hash tables.
    size: Hash table size (and feature dimension).
    units: MLP network parameters.
    clip: Minimum sigma value before enabling absorbance.
    _head: MLP output dimensionality (should always be 2).

    References
    ----------
    [1] Muller et al, "Instant Neural Graphics Primitives with a
        Multiresolution Hash Encoding," 2022.
    """

    _description = "NGP (instant Neural Graphics Primitive) architecture"

    def __init__(
            self, levels: Float32[Array, "n"], _head: int = 2,
            size: tuple[int, int] = (16384, 2), units: list[int] = [64, 32],
            clip: float = 0.1):
        super().__init__()
        self.size = size
        self.levels = levels
        self.clip = clip
        mlp: list[Callable] = []
        for u in units:
            mlp += [hk.Linear(u), jax.nn.leaky_relu]
        mlp.append(hk.Linear(_head))
        self.head = hk.Sequential(mlp)

    def hash(self, x: Integer[Array, "3"]) -> Integer[Array, ""]:
        """Apply hash function specified by NGP (Eq. 4 [1])."""
        ix = x.astype(jnp.uint32)
        pi2 = jnp.array(2654435761, dtype=jnp.uint32)
        pi3 = jnp.array(805459861, dtype=jnp.uint32)

        return (ix[0] + ix[1] * pi2 + ix[2] * pi3) % self.size[0]

    def lookup(self, x: Float32[Array, "3"]) -> Float32[Array, "n"]:
        """Multiresolution hash table lookup."""
        xscales = x.reshape(1, -1) * self.levels.reshape(-1, 1)
        grid = hk.get_parameter(
            "grid", (self.levels.shape[0], *self.size),
            init=hk.initializers.RandomUniform(0, 0.01))

        def interpolate_level(xscale, grid_level):
            def hash_table(c):
                return grid_level[self.hash(c)]
            return interpolate(xscale, jax.vmap(hash_table))

        return jax.vmap(interpolate_level)(xscales, grid)

    def alpha_clip(
        self, sigma: Float32[Array, ""], alpha: Float32[Array, ""]
    ) -> Float32[Array, ""]:
        """Alpha clipping function."""
        if self.clip > 0:
            return jnp.where(sigma > self.clip, jnp.minimum(alpha, 0.0), 0.0)
        else:
            return jnp.minimum(alpha, 0.0)

    def __call__(
        self, x: Float32[Array, "3"], dx: Optional[Float32[Array, "3"]] = None
    ) -> tuple[Float32[Array, ""], Float32[Array, ""]]:
        """Index into learned reflectance map."""
        table_out = self.lookup(x)
        sigma, alpha = self.head(table_out.reshape(-1))
        return sigma, self.alpha_clip(sigma, alpha)

    @classmethod
    def from_config(
        cls, levels=8, exponent=0.5, base=2, size=16, features=2,
        units=[64, 32], clip=0.1
    ) -> Callable[[], "NGP"]:
        """Create NGP haiku closure from config items."""
        def closure():
            return cls(
                levels=base * 2**(exponent * jnp.arange(levels)),
                size=(2**size, features), units=units, clip=clip)
        return closure

    @staticmethod
    def to_parser(p: types.ParserLike) -> None:
        """Create NGP command line arguments."""
        p.add_argument(
            "--levels", default=8, type=int, help="Hash table levels.")
        p.add_argument(
            "--exponent", default=0.43, type=float,
            help="Hash table level exponent, in powers of 2.")
        p.add_argument(
            "--base", default=2., type=float,
            help="Size of base (most coarse) hash table level.")
        p.add_argument(
            "--size", default=16, type=int,
            help="Hash table size, in powers of 2.")
        p.add_argument(
            "--features", default=2, type=int,
            help="Number of features per hash table level.")
        p.add_argument(
            "--units", default=[64, 32], nargs='+', type=int,
            help="Number of hidden units in the MLP head.")
        p.add_argument(
            "--clip", default=0.0, type=float,
            help="Sigma threshold before allowing absorbance (alpha < 0.0).")

    @staticmethod
    def args_to_config(args: types.Namespace) -> dict:
        """Create configuration dictionary."""
        return {
            "field_name": "NGP",
            "field": {
                "levels": args.levels, "exponent": args.exponent,
                "base": args.base, "size": args.size,
                "features": args.features, "units": args.units,
                "clip": args.clip
            }
        }


class NGPSH(NGP):
    """NGP [1] field with spherical harmonics [2].

    Parameters
    ----------
    harmonics: Number of spherical harmonic coefficients.
    kwargs: Passed to NGP.

    References
    ----------
    [1] Muller et al, "Instant Neural Graphics Primitives with a
        Multiresolution Hash Encoding," 2022.
    [2] Yu et al, "PlenOctrees For Real-time Rendering of Neural Radiance
        Fields," 2021.
    """

    _description = "NGP with view dependence using spherical harmonics"

    def __init__(self, harmonics: int = 25, **kwargs) -> None:
        assert harmonics in {1, 4, 9, 16, 25}
        self.harmonics = harmonics
        super().__init__(_head=harmonics + 1, **kwargs)

    def __call__(
        self, x: Float32[Array, "3"], dx: Optional[Float32[Array, "3"]] = None
    ) -> tuple[Float32[Array, ""], Float32[Array, ""]]:
        """Index into learned reflectance map."""
        table_out = self.lookup(x)
        mlp_out = self.head(table_out.reshape(-1))
        alpha = mlp_out[-1]

        if dx is None:
            sigma = mlp_out[0]
        else:
            sh = spherical_harmonics(dx, self.harmonics)
            sigma = jnp.sum(mlp_out[:-1] * sh)

        return sigma, self.alpha_clip(sigma, alpha)

    @classmethod
    def from_config(
        cls, levels=8, exponent=0.5, base=2, size=16, features=2,
        units=[64, 32], harmonics=25, clip=0.1
    ) -> Callable[[], "NGPSH"]:
        """Create NGP haiku closure from config items."""
        def closure():
            return cls(
                levels=base * 2**(exponent * jnp.arange(levels)),
                size=(2**size, features), units=units, harmonics=harmonics,
                clip=clip)
        return closure

    @staticmethod
    def to_parser(p: types.ParserLike) -> None:
        """Create NGP command line arguments."""
        p.add_argument(
            "--harmonics", default=25, type=int,
            help="Number of spherical harmonics.")
        NGP.to_parser(p)

    @staticmethod
    def args_to_config(args: types.Namespace) -> dict:
        """Create configuration dictionary."""
        cfg = NGP.args_to_config(args)
        cfg["field_name"] = "NGPSH"
        cfg["field"]["harmonics"] = args.harmonics
        return cfg
