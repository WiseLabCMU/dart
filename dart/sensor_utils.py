"""Front-end sensor utilities not used during training."""

from functools import partial
from jaxtyping import Float32, Integer, Array
from beartype.typing import Tuple, Callable, Optional
import json

import numpy as np
from jax import numpy as jnp
from jax import random, vmap

from .pose import RadarPose, sensor_to_world


class VirtualRadarUtils:
    """Mixin with various utilities."""

    def to_config(self, path: str = "data/sensor.json"):
        """Save radar parameters."""
        with open(path, 'w') as f:
            json.dump({
                "theta_lim": self.theta_lim,
                "phi_lim": self.phi_lim,
                "n": self.n,
                "k": self.k,
                "r": [float(x) for x in self.r],
                "d": [float(x) for x in self.d]
            }, f)

    @classmethod
    def from_config(
            cls, path: str = "data/sensor.json", **kwargs):
        """Load radar from saved parameters.

        Any keyword-args passed override the values in the config file.
        """
        with open(path) as f:
            cfg = json.load(f)
        cfg.update(kwargs)

        return cls(
            theta_lim=cfg["theta_lim"], phi_lim=cfg["phi_lim"],
            n=cfg["n"], k=cfg["k"],
            r=jnp.array(cfg["r"]).astype(jnp.float32),
            d=jnp.array(cfg["d"]).astype(jnp.float32)
        )

    def sample_points(
        self, key, r: Float32[Array, ""], d: Float32[Array, ""],
        pose: RadarPose
    ) -> Tuple[Float32[Array, "3 k"], Integer[Array, ""]]:
        """Sample points in world-space for the given (range, doppler) bin.

        Parameters
        ----------
        key: PRNGKey for random sampling.
        r, d: Range and doppler bins.
        pose: Sensor pose parameters.

        Returns
        -------
        points: Sampled points in sensor space.
        num_bins: Number of occupied bins (effective weight of samples).
        """
        psi = jnp.arange(self.n) * self.bin_width
        valid_psi = self.valid_mask(d, psi, pose)
        num_bins = jnp.sum(valid_psi)

        points_sensor = self.sample_rays(key, d, psi, valid_psi, pose)
        points_world = sensor_to_world(r, points_sensor, pose)
        return points_world, num_bins

    def render(
        self, key, sigma: Callable[[Float32[Array, "3"]], Float32[Array, ""]],
        pose: RadarPose
    ) -> Float32[Array, "nr nd"]:
        """Render single (range, doppler) radar image.

        NOTE: This function is not vmap or jit safe.

        Parameters
        ----------
        key: PRNGKey for random sampling.
        sigma: field function.
        pose: Sensor pose parameters.

        Returns
        -------
        Rendered image. Points not observed within the field of view are
        rendered as 0.
        """
        psi = jnp.arange(self.n) * self.bin_width
        valid_psi = vmap(partial(
            self.valid_mask, psi=psi, pose=pose))(self.d)
        num_bins = jnp.sum(valid_psi, axis=1)

        keys = jnp.array(random.split(key, self.d.shape[0]))

        t_sensor = vmap(partial(self.sample_rays, psi=psi, pose=pose))(
            keys, d=self.d, valid_psi=valid_psi)
        return vmap(
            partial(self.render_column, sigma=sigma, pose=pose)
        )(t_sensor, weight=num_bins.astype(float)).T
