"""Antenna gain characteristics."""

from jax import numpy as jnp

from jaxtyping import Float32, Array


def antenna_gain(
    theta: Float32[Array, "k"], phi: Float32[Array, "k"]
) -> Float32[Array, "k"]:
    """Compute single antenna gain for the TI AWR1843BOOST."""

    _theta = theta / jnp.pi * 180 / 56
    _phi = phi / jnp.pi * 180 / 56

    # Magic antenna gain function
    return jnp.exp((
        (0.14 * _phi**6 + 0.13 * _phi**4 - 8.2 * _phi**2)
        + (3.1 * _theta**8 - 22 * _theta**6 + 54 * _theta**4 - 55 * _theta**2)
    ).reshape(1, -1) / 10)
