"""Antenna gain characteristics."""

from jax import numpy as jnp
import jax

from jaxtyping import Float32, Array


def rect(
    theta: Float32[Array, "k"], phi: Float32[Array, "k"]
) -> Float32[Array, "1 k Na"]:
    """Rectangular gain (1 if within the field of view, 0 otherwise)."""
    return jnp.ones_like(theta).reshape(1, -1, 1)


def awr1843boost(
    theta: Float32[Array, "k"], phi: Float32[Array, "k"]
) -> Float32[Array, "1 k 1"]:
    """Compute single antenna gain for the TI AWR1843BOOST."""
    _theta = theta / jnp.pi * 180 / 56
    _phi = phi / jnp.pi * 180 / 56

    return jnp.power(10, (
        (0.14 * _phi**6 + 0.13 * _phi**4 - 8.2 * _phi**2)
        + (3.1 * _theta**8 - 22 * _theta**6 + 54 * _theta**4 - 55 * _theta**2)
    ).reshape(1, -1, 1) / 20)


def awr1843boost_az8(
    theta: Float32[Array, "k"], phi: Float32[Array, "k"]
) -> Float32[Array, "1 k 8"]:
    """Compute 8 antenna gains for 8 azimuth bins of the TI AWR1843BOOST."""
    gain = awr1843boost(theta, phi).reshape(-1)

    w = jnp.sin(-phi) * jnp.pi
    n = jnp.arange(8)

    def column(b):
        bin = (b / 8 * 2 - 1) * jnp.pi
        return gain * jnp.abs(
            jnp.sum(jnp.exp(jnp.outer(-1j * n, w - bin)), 0)) / 8

    return jax.vmap(column)(n).T.reshape(1, -1, 8)
