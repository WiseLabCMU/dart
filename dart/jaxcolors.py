"""JAX-native color conversions for GPU acceleration."""

from jax import numpy as jnp

from jaxtyping import Float, Array, Num
from dart import types


def hsv_to_rgb(
    hsv: Float[types.ArrayLike, "... 3"], _np=jnp
) -> Float[Array, "... 3"]:
    """Convert hsv values to rgb.

    Copied here, and modified for vectorization:
    https://matplotlib.org/3.1.1/_modules/matplotlib/colors.html#hsv_to_rgb
    and converted to jax.

    Parameters
    ----------
    hsv: HSV colors.
    _np: numpy-like backend to use.

    Returns
    -------
    RGB colors `float: (0, 1)`, using the array format corresponding to the
    provided backend.
    """
    in_shape = hsv.shape
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    i = (h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    r = sum((i % 6 == j) * x for j, x in enumerate([v, q, p, p, t, v, v]))
    g = sum((i % 6 == j) * x for j, x in enumerate([t, v, v, q, p, p, v]))
    b = sum((i % 6 == j) * x for j, x in enumerate([p, p, t, v, v, q, v]))

    rgb = jnp.stack([r, g, b], axis=-1)

    return rgb.reshape(in_shape)


def colormap(
    colors: Num[types.ArrayLike, "n d"],
    data: Float[types.ArrayLike, "..."],
    _np=jnp
) -> Num[Array, "... 3"]:
    """Apply a discrete colormap.
    
    Parameters
    ----------
    colors: list of discrete colors to apply (e.g. 0-255 RGB values). Can be
        an arbitrary number of channels, not just RGB.
    data: input data to index (`0 < data < 1`).
    _np: numpy-like backend to use.

    Returns
    -------
    An array with the same shape as `data`, with an extra dimension appended.
    """
    fidx = data * (colors.shape[0] - 1)
    left = _np.clip(_np.floor(fidx).astype(int), 0, colors.shape[0] - 1)
    return _np.take(colors, left, axis=0)
