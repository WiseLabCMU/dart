"""Doppler column methods."""

from jax import numpy as jnp
from jax import vmap

from jaxtyping import Float32, UInt8, Array
from beartype.typing import NamedTuple, Callable

from .pose import RadarPose, sensor_to_world


class TrainingColumn(NamedTuple):
    """Single column for training.

    For 256 range bins and 256 angular bins, this takes::

        96 + 256 / 8 + 4 + 4 = 136 bytes.

    Attributes
    ----------
    pose: pose for each column (96 bytes).
    valid: validity of each angular bin; bit-packed bool array (n / 8 bytes).
    weight: number of valid bins (4 bytes).
    doppler: doppler value for this column (4 bytes).
    """

    pose: RadarPose
    valid: UInt8[Array, "n8"]
    weight: Float32[Array, ""]
    doppler: Float32[Array, ""]


class VirtualRadarColumnMixins:
    """Radar doppler column methods."""

    def gain(self, t: Float32[Array, "3 k"]) -> Float32[Array, "k"]:
        """Compute antenna gain."""
        x, y, z = t
        theta = jnp.arcsin(z)
        phi = jnp.arcsin(y * jnp.cos(theta))
        _theta = theta / (2 * jnp.pi) * 180 / 56
        _phi = phi / (2 * jnp.pi) * 180 / 56

        return jnp.exp((
            (0.14 * _theta**6 + 0.13 * _theta**4 - 8.2 * _theta**2)
            + (3.1 * _phi**8 - 22 * _phi**6 + 54 * _phi**4 - 55 * _phi**2)
        ).reshape(1, -1) / 10)

    def render_column(
        self, t: Float32[Array, "3 k"],
        sigma: Callable[[Float32[Array, "3"]], Float32[Array, "2"]],
        pose: RadarPose, weight: Float32[Array, ""]
    ) -> Float32[Array, "nr"]:
        """Render a single doppler column for a radar image.

        Parameters
        ----------
        t: Sensor-space rays on the unit sphere.
        sigma: Field function.
        pose: Sensor pose.
        weight: Sample size weight.

        Returns
        -------
        Rendered column for one doppler value and a stack of range values.
        """
        def project_rays(r):
            t_world = sensor_to_world(r=r, t=t, pose=pose)
            return jnp.nan_to_num(vmap(sigma)(t_world.T))

        # Antenna Gain
        gain = self.gain(t)

        # Field steps
        field_vals = vmap(project_rays)(self.r)
        sigma_samples = field_vals[:, :, 0]
        alpha_samples = 1 - jnp.tanh(field_vals[:, :, 1])

        # Return signal
        transmitted = jnp.concatenate([
            jnp.ones((1, t.shape[1])),
            jnp.cumprod(alpha_samples[:-1], axis=0)
        ], axis=0)
        amplitude = sigma_samples * transmitted * gain

        constant = weight / self.n * self.r
        return jnp.mean(amplitude, axis=1) * constant

    def make_column(
        self, doppler: Float32[Array, ""], pose: RadarPose,
    ) -> TrainingColumn:
        """Create column for training.

        Parameters
        ----------
        d: doppler value.
        pose: sensor pose.

        Returns
        -------
        Training point with per-computed valid bins.
        """
        valid = self.valid_mask(doppler, pose)
        packed = jnp.packbits(valid)
        weight = jnp.sum(valid).astype(jnp.float32)
        return TrainingColumn(
            pose=pose, valid=packed, weight=weight, doppler=doppler)

    def column_forward(
        self, key, column: TrainingColumn,
        sigma: Callable[[Float32[Array, "3"]], Float32[Array, "2"]],
    ) -> Float32[Array, "nr"]:
        """Render a training column.

        Parameters
        ----------
        key : PRNGKey for random sampling.
        column: Pose and y_true.
        sigma: Field function.

        Returns
        -------
        Predicted doppler column.
        """
        valid = jnp.unpackbits(column.valid)
        t = self.sample_rays(
            key, d=column.doppler, valid_psi=valid, pose=column.pose)
        return self.render_column(
            t=t, sigma=sigma, pose=column.pose, weight=column.weight)
