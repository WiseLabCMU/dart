"""DART algorithm."""

from tqdm import tqdm as default_tqdm
from functools import partial
import pickle
import os

import jax
import numpy as np
from jax import numpy as jnp
import haiku as hk
import optax

from jaxtyping import Float32, Array, PyTree
from beartype.typing import Callable, NamedTuple, Optional, Union
from tensorflow.data import Dataset
from . import types

from .sensor import VirtualRadar
from .fields import NGP, NGPSH, SimpleGrid
from .utils import tf_to_jax, to_prngkey, update_avg
from .opt import sparse_adam
from .loss import get_loss_func


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
    loss: Loss function to use.
    weight: Function to apply before loss function.
    eps: Epsilon to use during training.
    """

    def __init__(
        self, sensor: VirtualRadar, optimizer: optax.GradientTransformation,
        sigma: Callable[[], types.SigmaField],
        project: Optional[Callable[[PyTree], PyTree]] = None,
        loss: Optional[types.LossFunc] = None
    ) -> None:

        def forward_train(batch: types.TrainingColumn):
            keys = jnp.array(
                jax.random.split(hk.next_rng_key(), batch.doppler.shape[0]))
            vfwd = jax.vmap(partial(sensor.column_forward, sigma=sigma()))
            return vfwd(keys, column=batch)

        def forward_test(batch: types.RadarPose):
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
        self.sensor = sensor
        self.loss = get_loss_func(eps=1e-4) if loss is None else loss

    def init(self, dataset, key: types.PRNGSeed = 42) -> ModelState:
        """Initialize model parameters and optimizer state."""
        sample = tf_to_jax(list(dataset.take(1))[0][0])
        params = self.model_train.init(to_prngkey(key), sample)
        opt_state = self.optimizer.init(params)
        return ModelState(params=params, opt_state=opt_state)

    def fit(
        self, train: Dataset, state: ModelState,
        val: Optional[Dataset] = None, epochs: int = 1,
        tqdm=default_tqdm, key: types.PRNGSeed = 42
    ) -> tuple[ModelState, list, list]:
        """Train model."""
        @jax.jit
        def loss_func(params, rng, batch):
            columns, y_true = batch
            y_pred = self.model_train.apply(params, rng, columns)
            return self.loss(y_pred, y_true)

        # Note: not putting step in a closure here (jitting grads and updates
        # separately) results in a ~100x performance penalty!
        @jax.jit
        def step(state, rng, batch):
            loss, grads = jax.value_and_grad(
                partial(loss_func, rng=rng, batch=batch))(state.params)

            clip = jax.tree_util.tree_map(jnp.nan_to_num, grads)
            updates, opt_state = self.optimizer.update(
                clip, state.opt_state, state.params)
            params = optax.apply_updates(state.params, updates)

            if self.project:
                params = self.project(params)
            return loss, ModelState(params, opt_state)

        train_log, val_log = [], []
        k = to_prngkey(key)
        for i in range(epochs):
            with tqdm(train, unit="batch", desc="Epoch {}".format(i)) as epoch:
                avg = 0.
                j = 0
                for _, batch in enumerate(epoch):
                    k, rng = jax.random.split(k, 2)
                    loss, _state = step(state, rng, tf_to_jax(batch))
                    if not jnp.isnan(loss):
                        state = _state
                        avg = update_avg(float(loss), avg, j, epoch)
                        j += 1
                    else:
                        print("Encountered NaN loss! Ignoring update.")
                train_log.append(avg)

            if val is not None:
                losses = []
                for j, batch in enumerate(epoch):
                    k, rng = jax.random.split(k, 2)
                    losses.append(
                        loss_func(state.params, rng, tf_to_jax(batch)))
                losses_jnp = jnp.array(losses)
                losses_jnp = losses_jnp[~jnp.isnan(losses_jnp)]
                val_loss = np.mean(losses_jnp)
                print("Val: {}".format(val_loss))
                val_log.append(float(val_loss))

        return state, train_log, val_log

    def save(self, path: str, state: ModelState) -> None:
        """Save state to file using pickle."""
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path: str) -> ModelState:
        """Load pickled state from file."""
        with open(path, 'rb') as f:
            return pickle.load(f)

    def render(
        self, params: Union[ModelState, PyTree], batch: types.RadarPose,
        key: types.PRNGSeed = 42
    ) -> Float32[Array, "w h b"]:
        """Render images from batch of poses."""
        if isinstance(params, ModelState):
            params = params.params
        return jax.jit(self.model.apply)(params, to_prngkey(key), batch)

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

    @classmethod
    def from_config(
        cls, sensor={}, field=None, lr=0.01, loss={}, **_
    ) -> "DART":
        """Create DART from config items."""
        return cls(
            VirtualRadar.from_config(**sensor),
            sparse_adam(lr=lr),          # optax.adam(0.01),
            NGPSH.from_config(**field),  # NGP.from_config(**field)
            loss=get_loss_func(**loss)
        )
