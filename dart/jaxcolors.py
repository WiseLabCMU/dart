"""JAX-native color conversions for GPU acceleration."""

from jax import numpy as jnp

from jaxtyping import Float32, Array


def hsv_to_rgb(hsv: Float32[Array, "... 3"]) -> Float32[Array, "... 3"]:
    """Convert hsv values to rgb.

    Copied here, and modified for vectorization:
    https://matplotlib.org/3.1.1/_modules/matplotlib/colors.html#hsv_to_rgb
    and converted to jax.
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
    colors: Float32[Array, "n 3"], data: Float32[Array, "..."]
) -> Float32[Array, "... 3"]:
    """Apply a discrete colormap."""
    fidx = data * (colors.shape[0] - 1)
    left = jnp.clip(jnp.floor(fidx).astype(int), 0, colors.shape[0] - 1)
    return jnp.take(colors, left, axis=0)
