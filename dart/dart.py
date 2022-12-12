"""DART algorithm."""

from tqdm import tqdm as default_tqdm
from functools import partial

import jax
from jax import numpy as jnp
import haiku as hk
import optax

from jaxtyping import Float32, Array, PyTree
from beartype.typing import Callable, NamedTuple, Optional

from .sensor import VirtualRadar


class ModelState(NamedTuple):
    """Model parameters and optimizer state."""

    params: PyTree
    opt_state: PyTree


class DART:
    """Doppler Aided Radar Tomography Algorithm.

    Parameters
    ----------
    sensor: Sensor model parameters.
    optimizer: Model optax optimizer.
    sigma: Field function generator; should close the actual field function.
    """

    def __init__(
        self, sensor: VirtualRadar, optimizer, sigma: Callable[
            [], Callable[[Float32[Array, "3"]], Float32[Array, ""]]]
    ) -> None:

        def forward(batch):
            key = hk.next_rng_key()
            keys = jnp.array(jax.random.split(key, batch.data.shape[0]))

            vfwd = jax.vmap(partial(sensor.column_forward, sigma=sigma()))
            return vfwd(keys, column=batch)

        self.model = hk.transform(forward)
        self.optimizer = optimizer

    def init(self, key, dataset) -> ModelState:
        """Initialize model parameters and optimizer state."""
        sample = jax.tree_util.tree_map(jnp.array, list(dataset.take(1))[0])
        params = self.model.init(key, sample)
        opt_state = self.optimizer.init(params)
        return ModelState(params=params, opt_state=opt_state)

    def fit(
        self, key, dataset, state: ModelState, epochs: int = 1,
        tqdm=default_tqdm
    ) -> ModelState:
        """Train model."""
        # Note: not putting step in a closure here results in a ~100x
        # performance penalty!
        def step(state, rng, batch):
            def loss_func(params):
                y_pred = self.model.apply(params, rng, batch)
                return jnp.sum((batch.data - y_pred)**2) / batch.data.shape[0]

            loss, grads = jax.value_and_grad(loss_func)(state.params)

            clip = jax.tree_util.tree_map(jnp.nan_to_num, grads)
            updates, opt_state = self.optimizer.update(
                clip, state.opt_state, state.params)
            params = optax.apply_updates(state.params, updates)
            # Todo: make this more general.
            params = jax.tree_util.tree_map(
                partial(jnp.clip, a_min=0.0, a_max=1.0), params)

            return loss, ModelState(params, opt_state)

        for i in range(epochs):
            with tqdm(
                dataset, unit="batch", desc="Epoch {}".format(i)
            ) as epoch:
                avg = 0.
                for j, batch in enumerate(epoch):
                    key, rng = jax.random.split(key, 2)
                    batch = jax.tree_util.tree_map(jnp.array, batch)

                    loss, state = jax.jit(step)(state, rng, batch)
                    avg = (avg * j + loss) / (j + 1)
                    epoch.set_postfix(loss=avg)

        return state
