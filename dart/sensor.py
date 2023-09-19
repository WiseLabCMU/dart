"""Ray sampling routines."""

from functools import partial

from jaxtyping import Float32, Bool, Array, Integer
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
    x, y, z = t
    theta = jnp.arcsin(jnp.clip(z, -0.99999, 0.99999))
    phi = jnp.arcsin(jnp.clip(y / jnp.cos(theta), -0.99999, 0.99999))
    return (theta, phi)


class VirtualRadar(NamedTuple):
    """Radar Sensor Model.

    Attributes
    ----------
    r, d: Range, doppler bins used for (r, d) images. Pass as (min, max, bins),
        i.e. the args of linspace in configuration (for `from_config`).
    n: Angular resolution; number of bins in a full circle of the
        (range sphere, doppler plane) intersection
    k: Sample size for stochastic integration
    gain: Antenna gain pattern.
    """

    r: Float32[Array, "Nr"]
    d: Float32[Array, "Nd"]
    n: int
    k: int
    gain: types.GainPattern

    @property
    def bin_width(self):
        """Alias for the width of bins in stochastic integration."""
        return 2 * jnp.pi / self.n

    @property
    def _extents(self):
        """Alias for the extents of a range-doppler plot."""
        return [self.d[0], self.d[-1], self.r[0], self.r[-1]]

    @classmethod
    def from_config(
        cls, n: int = 256, k: int = 128, r: list = [], d: list = [],
        gain: str = "awr1843boost"
    ) -> "VirtualRadar":
        """Create from configuration parameters."""
        return cls(
            r=jnp.linspace(*r), d=jnp.linspace(*d), n=n, k=k,
            gain=getattr(antenna, gain))

    def sample_rays(
            self, key, d: Float32[Array, ""],
            valid_psi: Bool[Array, "n"], pose: types.RadarPose
    ) -> Float32[Array, "3 k"]:
        """Sample rays according to pre-computed psi mask.

        Parameters
        ----------
        key: PRNGKey for random sampling.
        d: Doppler bin.
        valid_psi: Valid psi bins for angles jnp.arange(n) * bin_width in the
            (p, q) basis for the r-sphere d-plane intersection.
        pose: Sensor pose parameters.

        Returns
        -------
        Generated samples.
        """
        k1, k2 = random.split(key, 2)

        weights = valid_psi.astype(jnp.float32) * 10
        indices = random.categorical(k1, weights, shape=(self.k,))
        bin_centers = indices.astype(jnp.float32) * self.bin_width

        offsets = self.bin_width * (random.uniform(k2, shape=(self.k,)) - 0.5)
        psi_actual = bin_centers + offsets
        points = project_angle(d, psi_actual, pose)
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

        constant = weight / self.n
        return jnp.sum(amplitude, axis=1) * constant[..., jnp.newaxis]

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
        valid = jnp.unpackbits(column.valid).astype(bool)
        pose = adjust(column.pose)

        t = self.sample_rays(key, d=column.doppler, valid_psi=valid, pose=pose)
        return self._render_column(
            t=t, sigma=sigma, pose=pose, weight=column.weight)

    def valid_mask(
        self, d: Float32[Array, ""], pose: types.RadarPose
    ) -> Bool[Array, "n"]:
        """Get valid psi values within field of view as a mask.

        Computes a mask for bins::

            jnp.arange(n) * bin_width

        Parameters
        ----------
        d: Doppler bin.
        pose: Sensor pose parameters.

        Returns
        -------
        Output mask for each bin.
        """
        t = project_angle(d, jnp.arange(self.n) * self.bin_width, pose)
        return (t[0] > 0)

    def sample_points(
        self, key: types.PRNGKey, r: Float32[Array, ""],
        d: Float32[Array, ""], pose: types.RadarPose
    ) -> tuple[Float32[Array, "3 k"], Integer[Array, ""]]:
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
        valid_psi = self.valid_mask(d, pose)
        num_bins = jnp.sum(valid_psi)

        points_sensor = self.sample_rays(key, d, valid_psi, pose)
        points_world = sensor_to_world(r, points_sensor, pose)
        return points_world, num_bins

    def render(
        self, key: types.PRNGKey, sigma: types.SigmaField,
        pose: types.RadarPose
    ) -> Float32[Array, "Nr Nd Na"]:
        """Render single (range, doppler) radar image.

        NOTE: This function is not vmap or jit safe.

        Parameters
        ----------
        key: PRNGKey for random sampling.
        sigma: Field function.
        pose: Sensor pose parameters.

        Returns
        -------
        Rendered image. Points not observed within the field of view are
        rendered as 0.
        """
        valid_psi = vmap(partial(self.valid_mask, pose=pose))(self.d)
        weight = jnp.sum(valid_psi, axis=1).astype(jnp.float32) / pose.s

        keys = jnp.array(random.split(key, self.d.shape[0]))

        t_sensor: Float32[Array, "Nd 3"] = vmap(
            partial(self.sample_rays, pose=pose)
        )(keys, d=self.d, valid_psi=valid_psi)

        out: Float32[Array, "Nd Nr Na"] = vmap(
            partial(self._render_column, sigma=sigma, pose=pose)
        )(t_sensor, weight=weight)

        return jnp.swapaxes(out, 0, 1)
