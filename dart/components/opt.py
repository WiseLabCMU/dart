"""ADAM with updates only on nonzero gradients."""

import optax
import jax
from jax import numpy as jnp

from jaxtyping import Float32, Array, Int32, PyTree
from beartype.typing import NamedTuple


class ScaleByAdamState(NamedTuple):
    """Sparse adam state.

    Attributes
    ----------
    count: Count of how many times each parameter was updated; maintained
        individually per-parameter unlike normal Adam.
    mu: First moment exponential moving average.
    nu: Second moment exponential moving average.
    """

    count: PyTree[Int32[Array, "*"]]  # type: ignore
    mu: PyTree[Float32[Array, "*"]]
    nu: PyTree[Float32[Array, "*"]]


def _update_sparse(grads, old, new):
    return jax.tree_util.tree_map(
        lambda g, t1, t2: jnp.where(g != 0, t2, t1), grads, old, new)


def _safe_int32_increment(count):
    max_int32_value = jnp.iinfo(jnp.int32).max
    one = jnp.array(1, dtype=jnp.int32)
    return jax.tree_util.tree_map(
        lambda c: jnp.where(c < max_int32_value, c + one, max_int32_value),
        count)


def _bias_correction(moment, decay, count):
    return jax.tree_util.tree_map(
        lambda t, c: t / (1 - decay**c), moment, count)


def sparse_adam(
    lr: float = 0.01,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
) -> optax.GradientTransformation:
    """Create Sparse Adam Optimizer."""

    def init_fn(params):
        mu = jax.tree_util.tree_map(
            lambda t: jnp.zeros_like(t, dtype=jnp.float32), params)
        nu = jax.tree_util.tree_map(jnp.zeros_like, params)
        count = jax.tree_util.tree_map(
            lambda t: jnp.zeros_like(t, dtype=jnp.int32), params)
        return ScaleByAdamState(count=count, mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        count_inc = _update_sparse(
            updates, state.count, _safe_int32_increment(state.count))
        mu_new = _update_sparse(
            updates, state.mu, optax.update_moment(updates, state.mu, b1, 1))
        nu_new = _update_sparse(
            updates, state.nu,
            optax.update_moment_per_elem_norm(updates, state.nu, b2, 2))

        mu_hat = _bias_correction(mu_new, b1, count_inc)
        nu_hat = _bias_correction(nu_new, b2, count_inc)

        update = jax.tree_util.tree_map(
            lambda g, m, n: jnp.where(
                g != 0, -1 * lr * m / (jnp.sqrt(n + eps_root) + eps), 0),
            updates, mu_hat, nu_hat)

        return update, ScaleByAdamState(count=count_inc, mu=mu_new, nu=nu_new)

    return optax.GradientTransformation(init_fn, update_fn)
