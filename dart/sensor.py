"""Ray sampling routines."""

from functools import partial

from jaxtyping import Float32, Array
from beartype.typing import NamedTuple

from jax import numpy as jnp
from jax import random, vmap

from .components import antenna
from .adjustments import Adjustment
from . import types
from .pose import project_angle, sensor_to_world


def vec_to_angle(
    t: Float32[Array, "3 k"]
) -> tuple[Float32[Array, "k"], Float32[Array, "k"]]:
    """Get azimuth and elevation from unit sphere values."""
    _, y, z = t
    theta = jnp.arcsin(jnp.clip(z, -0.99999, 0.99999))
    phi = jnp.arcsin(jnp.clip(y / jnp.cos(theta), -0.99999, 0.99999))
    return (theta, phi)


class VirtualRadar(NamedTuple):
    """Radar Sensor Model.

    Attributes
    ----------
    r, d: Range, doppler bins used for (r, d) images. Pass as (min, max, bins),
        i.e. the args of linspace in configuration (for `from_config`).
    k: Sample size for stochastic integration
    gain: Antenna gain pattern.
    """

    r: Float32[Array, "Nr"]
    d: Float32[Array, "Nd"]
    k: int
    gain: types.GainPattern

    @property
    def _extents(self):
        """Alias for the extents of a range-doppler plot."""
        return [self.d[0], self.d[-1], self.r[0], self.r[-1]]

    @classmethod
    def from_config(
        cls, k: int = 128, r: list = [], d: list = [],
        gain: str = "awr1843boost"
    ) -> "VirtualRadar":
        """Create from configuration parameters."""
        return cls(
            r=jnp.linspace(*r), d=jnp.linspace(*d), k=k,
            gain=getattr(antenna, gain))
    
    @staticmethod
    def get_psi_min(
        d: Float32[Array, ""], pose: types.RadarPose
    ) -> Float32[Array, ""]:
        """Get psi value representing visible region of integration circle.

        Visible psi angles fall in the range of (-psi_min, psi_min). These
        angles all fall in front of the radar (+x).
        psi_min = pi means the entire circle is visible.
        psi_min = 0 means none of the circle is visible (behind the radar
        or speed is too low for this Doppler bin).
        
        Parameters
        ----------
        d: Doppler bin.
        pose: Sensor pose parameters.
        
        Returns
        -------
        psi_min angle in radians.
        """
        dnorm = d / pose.s
        vx = pose.v[0]
        h = vx * dnorm / jnp.sqrt(1 - vx * vx)
        r = jnp.sqrt(1 - dnorm * dnorm)
        psi_min = jnp.arccos(h / r)

        return jnp.where(
            (jnp.abs(dnorm) > 1) | (h > r),
            0,
            jnp.where(
                h < -r,
                jnp.pi,
                psi_min))
    
    def sample_rays(
        self, key: types.PRNGKey,
        d: Float32[Array, ""], pose: types.RadarPose
    ) -> Float32[Array, "3 k"]:
        """Sample rays according to pre-computed psi mask.

        Parameters
        ----------
        key : PRNGKey for random sampling.
        d: Doppler bin.
        pose: Sensor pose parameters.

        Returns
        -------
        Generated samples.
        """
        psi_min = self.get_psi_min(d, pose)
        #TODO fail if psi_min = 0, shouldn't happen for now with dataset
        delta_psi = 2 * psi_min / self.k
        psi = jnp.linspace(-psi_min, psi_min - delta_psi, self.k)
        psi += random.uniform(key) * delta_psi
        points = project_angle(d, psi, pose)
        return points

    def _render_column(
        self, t: Float32[Array, "3 k"], sigma: types.SigmaField,
        pose: types.RadarPose, weight: Float32[Array, ""]
    ) -> Float32[Array, "Nr Na"]:
        """Render a single doppler column for a radar image.

        Parameters
        ----------
        t: Sensor-space rays on the unit sphere.
        sigma: Field function.
        pose: Sensor pose.
        weight: Sample size weight.
        kwargs: Non-vectorized passthrough to sigma (i.e. hyperparams).

        Returns
        -------
        Rendered column for one doppler value and a stack of range values.
        """
        # Direction is the same for all ranges.
        dx = jnp.matmul(pose.A, t)

        def project_rays(r):
            t_world = sensor_to_world(r=r, t=t, pose=pose)
            return vmap(sigma)(t_world.T, dx=dx.T)

        sigma_samples, alpha_samples = vmap(project_rays)(self.r)

        # Return signal
        transmitted = jnp.concatenate([
            jnp.zeros((1, t.shape[1])),
            jnp.cumsum(alpha_samples[:-1], axis=0)
        ], axis=0)
        gain = self.gain(*vec_to_angle(t))
        amplitude: Float32[Array, "Nr k Na"] = (
            sigma_samples[..., jnp.newaxis] * gain
            * jnp.exp(transmitted)[..., jnp.newaxis])

        return jnp.sum(amplitude, axis=1) * weight[..., jnp.newaxis]

    def column_forward(
        self, key: types.PRNGKey, column: types.TrainingColumn,
        sigma: types.SigmaField, adjust: Adjustment
    ) -> Float32[Array, "Nr Na"]:
        """Render a training column.

        Parameters
        ----------
        key : PRNGKey for random sampling.
        column: Pose and y_true.
        sigma: Field function.
        adjust: Pose adjustment function.

        Returns
        -------
        Predicted doppler column.
        """
        pose = adjust(column.pose)

        t = self.sample_rays(key, d=column.doppler, pose=pose)
        return self._render_column(
            t=t, sigma=sigma, pose=pose, weight=column.weight)

    def render(
        self, key: types.PRNGKey, sigma: types.SigmaField,
        pose: types.RadarPose
    ) -> Float32[Array, "Nr Nd Na"]:
        """Render single (range, doppler) radar image.

        NOTE: This function is not vmap or jit safe.

        Parameters
        ----------
        key : PRNGKey for random sampling.
        sigma: Field function.
        pose: Sensor pose parameters.

        Returns
        -------
        Rendered image. Points not observed within the field of view are
        rendered as 0.
        """
        psi_min: Float32[Array, "Nd"] = vmap(
            partial(self.get_psi_min, pose=pose)
        )(d=self.d)
        weight = psi_min / jnp.pi / pose.s

        keys = jnp.array(random.split(key, self.d.shape[0]))

        t_sensor: Float32[Array, "Nd 3"] = vmap(
            partial(self.sample_rays, pose=pose)
        )(keys, d=self.d)

        out: Float32[Array, "Nd Nr Na"] = vmap(
            partial(self._render_column, sigma=sigma, pose=pose)
        )(t_sensor, weight=weight)

        return jnp.swapaxes(out, 0, 1)
