"""DART algorithm."""

from tqdm import tqdm as default_tqdm
from functools import partial
import os

import jax
import numpy as np
from jax import numpy as jnp
import haiku as hk
import optax

from jaxtyping import Float32, Array, PyTree
from beartype.typing import Callable, Optional, Union

from . import types
from . import fields, components, adjustments
from .sensor import VirtualRadar
from .camera import VirtualCamera, VirtualCameraImage
from .utils import (
    tf_to_jax, to_prngkey, update_avg, load_weights, save_weights)


class DART:
    """Doppler Aided Radar Tomography Algorithm.

    Parameters
    ----------
    sensor: Sensor model parameters.
    optimizer: Model optax optimizer.
    sigma: Field function closure.
    loss: Loss function to use.
    adjust: Pose adjustment model.
    schedules: Hyperparameter schedules to apply via kwargs to sigma.
    """

    def __init__(
        self, sensor: VirtualRadar,
        optimizer: optax.GradientTransformation,
        sigma: Callable[[], types.SigmaField],
        adjust: Callable[[], adjustments.Adjustment],
        loss: components.LossFunc,
        schedules: dict[str, types.HyperparameterSchedule] = {},
    ) -> None:

        def forward(batch: types.TrainingColumn, **kwargs):
            _adjust = adjust()

            keys = jnp.array(
                jax.random.split(hk.next_rng_key(), batch.doppler.shape[0]))
            vfwd = jax.vmap(partial(
                sensor.column_forward, sigma=partial(sigma(), **kwargs),
                adjust=_adjust))
            pred = vfwd(keys, column=batch)
            return pred, _adjust(None)

        self.sigma = sigma
        self.adjust = adjust
        self.model = hk.transform(forward)

        self.optimizer = optimizer
        self.sensor = sensor
        self.loss = loss
        self.schedules = schedules

    @classmethod
    def from_config(
        cls, sensor: dict = {}, field_name: str = "NGP", field: dict = {},
        adjustment_name: str = "Identity", adjustment: dict = {},
        lr: float = 0.01, loss: dict = {}, schedules: dict[str, dict] = {}, **_
    ) -> "DART":
        """Create DART from config items."""
        return cls(
            sensor=VirtualRadar.from_config(**sensor),
            optimizer=optax.adam(lr),
            sigma=getattr(fields, field_name).from_config(**field),
            adjust=getattr(
                adjustments, adjustment_name).from_config(**adjustment),
            loss=components.get_loss_func(**loss),
            schedules={
                k: getattr(components.schedules, v["func"])(**v["args"])
                for k, v in schedules.items()})

    def init(
        self, dataset: types.Dataset, key: types.PRNGSeed = 42
    ) -> types.ModelState:
        """Initialize model parameters and optimizer state."""
        sample = tf_to_jax(list(dataset.take(1))[0][0])
        params = self.model.init(to_prngkey(key), sample)
        opt_state = self.optimizer.init(params)
        return types.ModelState(params=params, opt_state=opt_state)

    def _hypers(
        self, epoch: int = -1, step: int = -1, train: bool = True
    ) -> dict[str, PyTree]:
        """Get hyperparameter schedule values."""
        res = {k: v(epoch, step) for k, v in self.schedules.items()}
        res.update({"epoch": epoch, "step": step})
        if not train:
            res["reg"] = 0.0
        return res

    def _train(
        self, step_func, tqdm, key: types.PRNGKey, state: types.ModelState,
        dataset: types.Dataset, epoch: int = 0, step: int = 0
    ) -> tuple[types.ModelState, float, int]:
        """Run training loop."""
        avg = types.Average(0.0, 0.0)
        with tqdm(dataset, unit="batch") as pbar:
            for batch in pbar:
                key, rng = jax.random.split(key, 2)
                hypers = self._hypers(epoch=epoch, step=step, train=True)
                loss, state = step_func(state, rng, tf_to_jax(batch), **hypers)
                avg = update_avg(float(loss), avg, pbar)
                if jnp.isnan(loss):
                    print("WARNING: encountered NaN loss!")
                step += 1
        return state, avg.avg, step

    def _val(
        self, loss_func, tqdm, key: types.PRNGKey, params: PyTree,
        dataset: types.Dataset, epoch: int = 0, step: int = 0
    ) -> float:
        """Run validation."""
        hypers = self._hypers(epoch=epoch, step=step, train=False)
        losses = []
        for batch in tqdm(dataset, unit="batch", desc="    Validating"):
            key, rng = jax.random.split(key, 2)
            losses.append(loss_func(params, rng, tf_to_jax(batch), **hypers))
        losses_np = np.array(losses)
        losses_np = losses_np[~np.isnan(losses_np)]
        loss_avg = float(np.mean(losses_np))
        print("Val: {}".format(loss_avg))
        return loss_avg

    def fit(
        self, train: types.Dataset, state: types.ModelState,
        val: Optional[types.Dataset] = None, epochs: int = 1,
        tqdm=default_tqdm, key: types.PRNGSeed = 42, save: Optional[str] = None
    ) -> tuple[types.ModelState, list, list]:
        """Train model."""
        @jax.jit
        def loss_func(params, rng, batch, **kwargs):
            columns, y_true = batch
            y_pred, reg = self.model.apply(params, rng, columns, **kwargs)
            return self.loss(y_pred, y_true) + reg * kwargs.get("reg", 1.0)

        # Note: not putting step in a closure here (jitting grads and updates
        # separately) results in a ~100x performance penalty!
        @jax.jit
        def step(state, rng, batch, **kwargs):
            loss, grads = jax.value_and_grad(
                partial(loss_func, rng=rng, batch=batch, **kwargs)
            )(state.params)

            clip = jax.tree_util.tree_map(jnp.nan_to_num, grads)
            updates, opt_state = self.optimizer.update(
                clip, state.opt_state, state.params)
            params = optax.apply_updates(state.params, updates)

            return loss, types.ModelState(params, opt_state)

        train_log, val_log = [], []
        stepidx = 0
        k = to_prngkey(key)

        k, rng = jax.random.split(k, 2)
        self._val(loss_func, tqdm, rng, state.params, val)

        for i in range(epochs):
            print("Schedule:", self._hypers(epoch=i, step=stepidx))
            try:
                k, k1, k2 = jax.random.split(k, 3)
                pbar = partial(tqdm, desc="Epoch {}".format(i))
                state, loss, stepidx = self._train(
                    step, pbar, k1, state, train, epoch=i, step=stepidx)
                train_log.append(float(loss))

                if val is not None:
                    val_log.append(float(self._val(
                        loss_func, tqdm, k2, state.params, val,
                        epoch=i, step=stepidx)))
                if save is not None:
                    self.save("{}_{}".format(save, i), state)
            except KeyboardInterrupt:
                break

        return state, train_log, val_log

    def save(self, path: str, state: types.ModelState) -> None:
        """Save state to file using pickle."""
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        save_weights(state.params, path)

    def load(self, path: str) -> dict:
        """Load pickled state from file."""
        return load_weights(path)

    def render(
        self, params: Union[types.ModelState, PyTree], batch: types.RadarPose,
        key: types.PRNGSeed = 42
    ) -> Float32[Array, "w h b a"]:
        """Render images from batch of poses."""
        def forward(batch: types.RadarPose):
            keys = jnp.array(
                jax.random.split(hk.next_rng_key(), batch.x.shape[0]))
            vfwd = jax.vmap(partial(
                self.sensor.render,
                sigma=partial(self.sigma(), **self._hypers())))
            return vfwd(keys, pose=batch)

        return hk.transform(forward).apply(
            types.ModelState.get_params(params), to_prngkey(key), batch)

    def grid(
        self, params: Union[types.ModelState, PyTree],
        x: Float32[Array, "x"], y: Float32[Array, "y"], z: Float32[Array, "z"],
        key: types.PRNGSeed = 42
    ) -> tuple[Float32[Array, "x y z"], Float32[Array, "x y z"]]:
        """Evaluate model as a fixed grid."""
        def forward_grid(batch: Float32[Array, "n 3"]):
            return jax.vmap(partial(self.sigma(), **self._hypers()))(batch)

        grid = jnp.meshgrid(x, y, z, indexing='ij')
        xyz = jnp.stack(grid, axis=-1).reshape(-1, 3)
        sigma, alpha = hk.transform(forward_grid).apply(
            types.ModelState.get_params(params), to_prngkey(key), xyz)
        shape = (x.shape[0], y.shape[0], z.shape[0])
        return sigma.reshape(shape), alpha.reshape(shape)

    def camera(
        self, params: Union[types.ModelState, PyTree], batch: types.RadarPose,
        camera: VirtualCamera, key: types.PRNGSeed = 42
    ) -> VirtualCameraImage:
        """Render camera images."""
        def forward(batch: types.RadarPose):
            vfwd = jax.vmap(partial(
                camera.render, field=partial(self.sigma(), **self._hypers())))
            return vfwd(pose=batch)

        return hk.transform(forward).apply(
            types.ModelState.get_params(params), to_prngkey(key), batch)
