"""Instant Neural Graphics Primitive (NGP) based fields.

References
----------
[1] Muller et al, "Instant Neural Graphics Primitives with a
    Multiresolution Hash Encoding," 2022.
[2] Yu et al, "PlenOctrees For Real-time Rendering of Neural Radiance
    Fields," 2021.
[3] Nerfacto. https://docs.nerf.studio/nerfology/methods/nerfacto.html
"""

from jax import numpy as jnp
import jax
import haiku as hk

from jaxtyping import Float32, Integer, Array, Float
from beartype.typing import Optional, Callable

from dart import types
from ._spatial import interpolate, spherical_harmonics


def _clip(x):
    return jnp.minimum(0.0, x)

@jax.custom_vjp
def clip(x):
    """Gradient estimator for the transmittance activation `min(0, alpha)`.

    At a high level, we don't pass through gradients if we are already
    clipping alpha and the gradients want to push alpha even more positive.

    Specifically:
      - `g < 0` indicates alpha will decrease the loss. If `alpha > 0`, then
        this won't be productive, so we don't pass the grads.
      - Otherwise, we use a straight-through estimator (i.e. if `alpha <= 0`,
        or if `alpha > 0` is being clipped, but the gradient is trying to
        un-clip it).
    """
    return _clip(x)

def _clip_fwd(x):
    return _clip(x), x

def _clip_bwd(res, g):
    grad = jnp.where((res > 0) & (g < 0), 0.0, g)
    return (grad,)

clip.defvjp(_clip_fwd, _clip_bwd)


class NGP(hk.Module):
    """NGP field [1] with sigma (reflectance) and alpha (transmittance) output.

    Parameters
    ----------
    levels: Resolution of each hash table level. The length determines the
        number of hash tables.
    size: Hash table size (and feature dimension).
    units: MLP network parameters.
    alpha_scale: Transmittance scale factor (for initialization stability).
    _head: MLP output dimensionality (should always be 2).
    """

    _description = "NGP (instant Neural Graphics Primitive) architecture"

    def __init__(
            self, levels: Float32[Array, "n"], _head: int = 2,
            size: tuple[int, int] = (16384, 2), units: list[int] = [64, 32],
            alpha_scale: float = 0.1):
        super().__init__()
        self.size = size
        self.levels = levels
        self.alpha_scale = alpha_scale
        mlp: list[Callable] = []
        for u in units:
            mlp += [hk.Linear(u), jax.nn.leaky_relu]  # type: ignore
        mlp.append(hk.Linear(_head))  # type: ignore
        self.head = hk.Sequential(mlp)  # type: ignore

    def hash(self, x: Integer[Array, "3"]) -> Integer[Array, ""]:
        """Apply hash function specified by NGP (Eq. 4 [1])."""
        ix = x.astype(jnp.uint32) - (x < 0).astype(jnp.uint32)
        pi2 = jnp.array(2654435761, dtype=jnp.uint32)
        pi3 = jnp.array(805459861, dtype=jnp.uint32)

        return (ix[0] ^ (ix[1] * pi2) ^ (ix[2] * pi3)) % self.size[0]

    def lookup(self, x: Float32[Array, "3"]) -> Float32[Array, "n d"]:
        """Multiresolution hash table lookup."""
        xscales = (
            x[None, :] * self.levels[:, None]
            + jnp.arange(self.levels.shape[0])[:, None])
        grid = hk.get_parameter(
            "grid", (self.levels.shape[0], *self.size),
            init=hk.initializers.RandomUniform(0, 0.01))

        def interpolate_level(xscale, grid_level):
            def hash_table(c):
                return grid_level[self.hash(c)]
            return interpolate(xscale, jax.vmap(hash_table))

        return jax.vmap(interpolate_level)(xscales, grid)

    def _activation(
        self, sigma: Float[Array, ""], alpha: Float[Array, ""],
        alpha_clip: Optional[types.FloatLike] = None
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        """Apply clipping activation function."""
        if alpha_clip is not None:
            alpha = jnp.where(sigma > alpha_clip, alpha, 0)

        return sigma, clip(alpha) * self.alpha_scale

    def __call__(
        self, x: Float32[Array, "3"], dx: Optional[Float32[Array, "3"]] = None,
        alpha_clip: Optional[types.FloatLike] = None, **kwargs
    ) -> tuple[Float32[Array, ""], Float32[Array, ""]]:
        """Index into learned reflectance map."""
        table_out = self.lookup(x)
        sigma, alpha = self.head(table_out.reshape(-1))
        return self._activation(sigma, alpha, alpha_clip=alpha_clip)

    @classmethod
    def from_config(
        cls, levels=8, exponent=0.5, base=2, size=16, features=2, **kwargs
    ) -> Callable[[], "NGP"]:
        """Create NGP haiku closure from config items."""
        def closure():
            return cls(  # type: ignore
                levels=base * 2**(exponent * jnp.arange(levels)),
                size=(2**size, features), **kwargs)
        return closure

    @staticmethod
    def to_parser(p: types.ParserLike) -> None:
        """Create NGP command line arguments."""
        p.add_argument(
            "--levels", default=12, type=int, help="Hash table levels.")
        p.add_argument(
            "--exponent", default=0.43, type=float,
            help="Hash table level exponent, in powers of 2.")
        p.add_argument(
            "--base", default=4., type=float,
            help="Size of base (most coarse) hash table level.")
        p.add_argument(
            "--size", default=20, type=int,
            help="Hash table size, in powers of 2.")
        p.add_argument(
            "--features", default=2, type=int,
            help="Number of features per hash table level.")
        p.add_argument(
            "--units", default=[64, 32], nargs='+', type=int,
            help="Number of hidden units in the MLP head.")
        p.add_argument(
            "--alpha_scale", default=0.1, type=float,
            help="Transmittance scale factor for intialization stability.")

    @staticmethod
    def args_to_config(args: types.Namespace) -> dict:
        """Create configuration dictionary."""
        return {
            "field_name": "NGP",
            "field": {
                "levels": args.levels, "exponent": args.exponent,
                "base": args.base, "size": args.size,
                "features": args.features, "units": args.units,
                "alpha_scale": args.alpha_scale
            }
        }


class NGPSH(NGP):
    """NGP [1] field with plenoctrees-style spherical harmonics [2].

    Has two different behaviors:
     1. Ray-tracing mode (dx=float[3]): apply spherical harmonic coefficients
        to sigma and alpha.
     2. Map mode (dx=None): return ||coef||_2 for sigma and alpha. This
        corresponds to the L2 norm of sigma and alpha over the sphere.

    Parameters
    ----------
    harmonics: Number of spherical harmonic coefficients.
    kwargs: Passed to NGP.
    """

    _description = "NGP with plenoctrees-style view dependence"

    def __init__(self, harmonics: int = 25, **kwargs) -> None:
        assert harmonics in {1, 4, 9, 16, 25}
        self.harmonics = harmonics
        super().__init__(_head=harmonics + 2, **kwargs)

    def __call__(
        self, x: Float32[Array, "3"], dx: Optional[Float32[Array, "3"]] = None,
        alpha_clip: Optional[types.FloatLike] = None, **kwargs
    ) -> tuple[Float32[Array, ""], Float32[Array, ""]]:
        """Index into learned reflectance map.
        
        Parameters
        ----------
        x, dx: Query position and orientation.
        alpha_clip: Alpha clipping threshold to eliminate false "radar
            absorbing stealth composites".
        """
        table_out = self.lookup(x)
        mlp_out = self.head(table_out.reshape(-1))
        sigma, alpha = mlp_out[:2]

        if dx is not None:
            sh = spherical_harmonics(dx, self.harmonics)
            components = mlp_out[2:] / jnp.linalg.norm(mlp_out[2:], ord=2)
            proj = jnp.sum(components * sh)
            sigma = sigma * jnp.abs(proj)
            alpha = alpha * jnp.abs(proj)

        return self._activation(sigma, alpha, alpha_clip=alpha_clip)

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


class NGPSH2(NGP):
    """NGP [1] field with neural network-style view dependence [3]."""

    _description = "NGP with nerfacto-style view dependence"

    def __init__(self, harmonics: int = 16, **kwargs) -> None:
        assert harmonics in {1, 4, 9, 16, 25}
        self.harmonics = harmonics
        super().__init__(_head=2, **kwargs)

    def __call__(
        self, x: Float32[Array, "3"], dx: Optional[Float32[Array, "3"]] = None,
        alpha_clip: Optional[types.FloatLike] = None, **kwargs
    ) -> tuple[Float32[Array, ""], Float32[Array, ""]]:
        """Index into learned reflectance map."""
        table_out = self.lookup(x)
        sh = (
            spherical_harmonics(dx, self.harmonics) if dx is not None
            else jnp.zeros(self.harmonics, dtype=jnp.float32))

        mlp_out = self.head(jnp.concatenate([table_out.reshape(-1), sh]))
        sigma, alpha = mlp_out

        return self._activation(sigma, alpha, alpha_clip=alpha_clip)

    @staticmethod
    def to_parser(p: types.ParserLike) -> None:
        """Create NGP command line arguments."""
        p.add_argument(
            "--levels", default=12, type=int, help="Hash table levels.")
        p.add_argument(
            "--exponent", default=0.43, type=float,
            help="Hash table level exponent, in powers of 2.")
        p.add_argument(
            "--base", default=4., type=float,
            help="Size of base (most coarse) hash table level.")
        p.add_argument(
            "--size", default=20, type=int,
            help="Hash table size, in powers of 2.")
        p.add_argument(
            "--features", default=2, type=int,
            help="Number of features per hash table level.")
        p.add_argument(
            "--units", default=[64], nargs='+', type=int,
            help="Number of hidden units in the MLP head.")
        p.add_argument(
            "--alpha_scale", default=0.1, type=float,
            help="Transmittance scale factor for intialization stability.")
        p.add_argument(
            "--harmonics", default=16, type=int,
            help="Number of spherical harmonics.")

    @staticmethod
    def args_to_config(args: types.Namespace) -> dict:
        """Create configuration dictionary."""
        cfg = NGP.args_to_config(args)
        cfg["field_name"] = "NGPSH2"
        cfg["field"]["harmonics"] = args.harmonics
        return cfg
