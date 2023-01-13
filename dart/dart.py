"""DART algorithm."""

from tqdm import tqdm as default_tqdm
from functools import partial
import pickle

import jax
from jax import numpy as jnp
import haiku as hk
import optax

from jaxtyping import Float32, Array, PyTree, Integer
from beartype.typing import Callable, NamedTuple, Optional, Union

from .pose import RadarPose
from .sensor import VirtualRadar
from .sensor_column import TrainingColumn


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
    project: Projection if the field requires projected gradient descent.
    """

    def __init__(
        self, sensor: VirtualRadar, optimizer: optax.GradientTransformation,
        sigma: Callable[
            [], Callable[[Float32[Array, "3"]], Float32[Array, "2"]]],
        project: Optional[Callable[[PyTree], PyTree]] = None
    ) -> None:

        def forward_train(batch: TrainingColumn):
            keys = jnp.array(
                jax.random.split(hk.next_rng_key(), batch.doppler.shape[0]))
            vfwd = jax.vmap(partial(sensor.column_forward, sigma=sigma()))
            return vfwd(keys, column=batch)

        def forward_test(batch: RadarPose):
            keys = jnp.array(
                jax.random.split(hk.next_rng_key(), batch.x.shape[0]))
            vfwd = jax.vmap(partial(sensor.render, sigma=sigma()))
            return vfwd(keys, pose=batch)

        def forward_grid(batch: Float32[Array, "n 2"]):
            return jax.vmap(sigma())(batch)

        self.model_train = hk.transform(forward_train)
        self.model = hk.transform(forward_test)
        self.model_grid = hk.transform(forward_grid)
        self.optimizer = optimizer
        self.project = project

    def init(self, key, dataset) -> ModelState:
        """Initialize model parameters and optimizer state."""
        sample = jax.tree_util.tree_map(jnp.array, list(dataset.take(1))[0][0])
        params = self.model_train.init(key, sample)
        opt_state = self.optimizer.init(params)
        return ModelState(params=params, opt_state=opt_state)

    def fit(
        self, key, dataset, state: ModelState, epochs: int = 1,
        tqdm=default_tqdm
    ) -> ModelState:
        """Train model."""
        # Note: not putting step in a closure here results in a ~100x
        # performance penalty!
        def step(state, rng, columns, y_true):
            def loss_func(params):
                y_pred = self.model_train.apply(params, rng, columns)
                return jnp.sum((y_true - y_pred)**2) / y_true.shape[0]

            loss, grads = jax.value_and_grad(loss_func)(state.params)

            clip = jax.tree_util.tree_map(jnp.nan_to_num, grads)
            updates, opt_state = self.optimizer.update(
                clip, state.opt_state, state.params)
            params = optax.apply_updates(state.params, updates)

            if self.project:
                params = self.project(params)
            return loss, ModelState(params, opt_state)

        for i in range(epochs):
            with tqdm(
                dataset, unit="batch", desc="Epoch {}".format(i)
            ) as epoch:
                avg = 0.
                for j, batch in enumerate(epoch):
                    key, rng = jax.random.split(key, 2)
                    columns, y_true = jax.tree_util.tree_map(jnp.array, batch)

                    loss, state = jax.jit(step)(state, rng, columns, y_true)
                    avg = (avg * j + loss) / (j + 1)
                    epoch.set_postfix(loss=avg)

        return state

    def save(self, path: str, state: ModelState) -> None:
        """Save state to file using pickle."""
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path: str) -> ModelState:
        """Load pickled state from file."""
        with open(path, 'rb') as f:
            return pickle.load(f)

    def render(
        self, params: Union[ModelState, PyTree], batch: RadarPose,
        key: Union[Integer[Array, "2"], int] = 42
    ) -> Float32[Array, "w h b"]:
        """Render images from batch of poses."""
        if isinstance(params, ModelState):
            params = params.params
        if isinstance(key, int):
            key = jax.random.PRNGKey(key)
        return jax.jit(self.model.apply)(params, key, batch)

    def grid(
        self, params: Union[ModelState, PyTree],
        x: Float32[Array, "x"], y: Float32[Array, "y"], z: Float32[Array, "z"]
    ) -> Float32[Array, "x y z"]:
        """Evaluate model as a fixed grid."""
        if isinstance(params, ModelState):
            params = params.params
        xyz = jnp.stack(
            jnp.meshgrid(x, y, z, indexing='ij'), axis=-1).reshape(-1, 3)
        values = jax.jit(self.model_grid.apply)(params, None, xyz)
        return values.reshape(x.shape[0], y.shape[0], z.shape[0], 2)
