"""Spatial utilities."""

from jaxtyping import Float32, Integer, Array, Float
from beartype.typing import Union, Callable

from jax import numpy as jnp


def interpolate(
    x: Float32[Array, "3"],
    grid: Union[
        Float[Array, "nx ny nz d"],
        Callable[[Integer[Array, "8 3"]], Float[Array, "8 d"]]
    ]
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
    ) != 0).astype(jnp.int32)
    # (0, 0, 0) and (1, 1, 1) cube bounds
    bounds = jnp.stack([jnp.floor(x), jnp.ceil(x)]).astype(jnp.int32)
    # Corner values
    c = bounds[mask, [0, 1, 2]]
    if isinstance(grid, jnp.ndarray):
        values = grid[c[:, 0], c[:, 1], c[:, 2]].reshape(8, -1)
    else:
        values = grid(c).reshape(8, -1)
    # Distances from opposite corner (same coord -> make weight 1)
    dist = jnp.abs(bounds[1 - mask, [0, 1, 2]].astype(float) - x)
    dist += (dist == 0)
    # Opposite volume (https://en.wikipedia.org/wiki/Trilinear_interpolation)
    weights = jnp.prod(dist, axis=1).reshape(8, -1)
    return jnp.sum(values * weights / jnp.sum(weights), axis=0)


def spherical_harmonics(
    dx: Float32[Array, "3"], harmonics: int = 25
) -> Float32[Array, "harmonics"]:
    """Compute spherical harmonic coefficients.

    Parameters
    ----------
    dx: offset direction from the point to the sensor (on the unit sphere).
    harmonics: number of coefficients to compute.

    References
    ----------
    https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
        *NOTE*: make sure to use the real spherical harmonics, not the complex
        spherical harmonics.
    https://github.com/nerfstudio-project/nerfstudio:
        Implementation here: nerfstudio/utils/math.py
        With bugfix applied:
        https://github.com/nerfstudio-project/nerfstudio/issues/2081
    """
    x, y, z = dx
    xx = x**2
    yy = y**2
    zz = z**2

    coef = [0.28209479177387814]
    if harmonics >= 4:
        coef += [
            0.4886025119029199 * y,
            0.4886025119029199 * z,
            0.4886025119029199 * x]
    if harmonics >= 9:
        coef += [
            1.0925484305920792 * x * y,
            1.0925484305920792 * y * z,
            0.9461746957575601 * zz - 0.31539156525251999,
            1.0925484305920792 * x * z,
            0.5462742152960396 * (xx - yy)]
    if harmonics >= 16:
        coef += [
            0.5900435899266435 * y * (3 * xx - yy),
            2.890611442640554 * x * y * z,
            0.4570457994644658 * y * (5 * zz - 1),
            0.3731763325901154 * z * (5 * zz - 3),
            0.4570457994644658 * x * (5 * zz - 1),
            1.445305721320277 * z * (xx - yy),
            0.5900435899266435 * x * (xx - 3 * yy)]
    if harmonics >= 25:
        coef += [
            2.5033429417967046 * x * y * (xx - yy),
            1.7701307697799304 * y * z * (3 * xx - yy),
            0.9461746957575601 * x * y * (7 * zz - 1),
            0.6690465435572892 * y * z * (7 * zz - 3),
            0.10578554691520431 * (35 * zz * zz - 30 * zz + 3),
            0.6690465435572892 * x * z * (7 * zz - 3),
            0.47308734787878004 * (xx - yy) * (7 * zz - 1),
            1.7701307697799304 * x * z * (xx - 3 * yy),
            0.6258357354491761 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))]

    return jnp.array(coef)
