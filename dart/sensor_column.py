"""Doppler column methods."""

from jax import numpy as jnp
from jax import vmap

from jaxtyping import Float32, UInt8, Array, Bool
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

    def render_column(
        self, t: Float32[Array, "3 k"],
        sigma: Callable[[Float32[Array, "3"]], Float32[Array, ""]],
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

        sigma_samples = vmap(project_rays)(self.r)
        energy = jnp.cumprod(1 - sigma_samples[:-1], axis=0)
        reflected = energy * sigma_samples[1:]

        return (
            jnp.concatenate([
                jnp.array(jnp.mean(sigma_samples[0])).reshape((1,)),
                jnp.mean(reflected, axis=1)])
            * 2 * jnp.pi * self.r * weight / self.n)

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
        sigma: Callable[[Float32[Array, "3"]], Float32[Array, ""]],
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
