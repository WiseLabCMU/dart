"""Ray sampling routines.

Conventions
-----------
- The sensor field of view is centered around +x.
- +y is left of +x.
- +z is straight up.
"""

from functools import partial
from jaxtyping import Float32, Integer, Bool, Array, jaxtyped
from beartype import beartype as typechecker
from beartype.typing import NamedTuple, Tuple, Callable, Optional

from jax import numpy as jnp
from jax import random, vmap


class RadarPose(NamedTuple):
    """Radar pose parameters.

    Attributes
    ----------
    v: normalized velocity direction (``||v||_2=1``).
    p, q: orthonormal basis along with ``v``.
    s: speed (magnitude of un-normalized velocity).
    x: sensor location in global coordinates.
    A: 3D rotation matrix for sensor pose; should transform sensor-space to
        world-space.
    """

    v: Float32[Array, "3"]
    p: Float32[Array, "3"]
    q: Float32[Array, "3"]
    s: Float32[Array, "3"]
    x: Float32[Array, "3"]
    A: Float32[Array, "3 3"]


@jaxtyped
@typechecker
class VirtualRadar:
    """Virtual Radar Sensor Model.

    Parameters
    ----------
    r, d: Range and Doppler bins used for (r, d) images.
    theta_lim, phi_lim: Bounds (radians) on elevation and azimuth angle;
        +/- pi/12 (15 degrees) and pi/3 (60 degrees) by default.
    n: Angular resolution; number of bins in a full circle of the
        (range sphere, doppler plane) intersection
    k: Sample size for stochastic integration
    """

    def __init__(
        self, theta_lim: float = jnp.pi / 12, phi_lim: float = jnp.pi / 3,
        n: int = 360, k: int = 120,
        r: Optional[Float32[Array, "nr"]] = None,
        d: Optional[Float32[Array, "nd"]] = None,
    ) -> None:
        self.r = r
        self.d = d
        self.theta_lim = theta_lim
        self.phi_lim = phi_lim
        self.n = n
        self.k = k
        self.bin_width = 2 * jnp.pi / n

    @staticmethod
    def make_pose(
        v: Float32[Array, "3"], x: Float32[Array, "3"],
        A: Float32[Array, "3 3"],
    ) -> RadarPose:
        """Create pose data namedtuple.

        Parameters
        ----------
        v: Velocity vector in global coordinates.
        x: Sensor location in global coordinates.
        A: 3D rotation matrix for sensor pose; should transform sensor-space to
            world-space.

        Returns
        -------
        Created pose object.
        """
        # Transform velocity to sensor space and separate magnitude
        v_sensor = jnp.matmul(jnp.linalg.inv(A), v)
        s = jnp.linalg.norm(v_sensor)
        v = v_sensor / s

        # This takes an identity matrix, mods out v, and turns the remainder
        # into an orthonormal basis using SVD for best stability.
        _, _, _V = jnp.linalg.svd(jnp.eye(3) - jnp.outer(v, v))
        p, q = _V[:2]

        return RadarPose(v=v, s=s, p=p, q=q, x=x, A=A)

    @staticmethod
    def project(
        d: Float32[Array, ""], psi: Float32[Array, "n"], pose: RadarPose
    ) -> Float32[Array, "3 n"]:
        """Generate projections on a unit sphere.

        Parameters
        ----------
        d: Doppler bin.
        psi: Angles to project on the doppler-sphere intersection.
        pose: Sensor pose parameters.

        Returns
        -------
        Projected (x, y, z) coordinates.
        """
        d_norm = d / pose.s
        return (
            jnp.sqrt(1 - d_norm**2) * (
                jnp.outer(pose.p, jnp.cos(psi))
                + jnp.outer(pose.q, jnp.sin(psi)))
            + pose.v.reshape(3, 1) * d_norm)

    def valid_mask(
        self, d: Float32[Array, ""], psi: Float32[Array, "n"], pose: RadarPose
    ) -> Bool[Array, "n"]:
        """Get valid psi values within field of view as a mask.

        Parameters
        ----------
        d: Doppler bin.
        psi: Angles to check on the doppler-sphere intersection.
        pose: Sensor pose parameters.

        Returns
        -------
        Output mask for each bin.
        """
        x, y, z = self.project(d, psi, pose)

        theta = jnp.arcsin(z)
        phi = jnp.arcsin(y * jnp.cos(theta))
        return (
            (theta < self.theta_lim) & (theta > -self.theta_lim)
            & (phi < self.phi_lim) & (phi > -self.phi_lim)
            & (x > 0))

    def sample_rays(
            self, key, d: Float32[Array, ""], psi: Float32[Array, "n"],
            valid_psi: Bool[Array, "n"], pose: RadarPose
    ) -> Float32[Array, "3 k"]:
        """Sample rays according to pre-computed psi.

        Parameters
        ----------
        key: PRNGKey for random sampling.
        d: Doppler bin.
        psi: Angles in the (p, q) basis for the r-sphere d-plane intersection.
        valid_psi: Valid psi bins.
        pose: Sensor pose parameters.

        Returns
        -------
        Generated samples.
        """
        k1, k2 = random.split(key, 2)

        weights = valid_psi.astype(jnp.float32) * 10
        indices = random.categorical(k1, weights, shape=(self.k,))
        bin_centers = psi[indices]

        offsets = self.bin_width * (random.uniform(k2, shape=(self.k,)) - 0.5)
        psi_actual = bin_centers + offsets
        points = self.project(d, psi_actual, pose)
        return points

    @staticmethod
    def sensor_to_world(
        r: Float32[Array, ""], x: Float32[Array, "3 k"], pose: RadarPose
    ) -> Float32[Array, "3 k"]:
        """Project points to world-space.

        Parameters
        ----------
        r: Range bin.
        x: Positions in unit-sphere sensor-space.
        pose: Sensor pose.

        Returns
        -------
        Projected points at the specified range in world-space.
        """
        return pose.x.reshape(3, 1) + jnp.matmul(pose.A, r * x)

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
        points_world = self.sensor_to_world(r, points_sensor, pose)
        return points_world, num_bins

    def render_column(
        self, t_sensor: Float32[Array, "3 k"],
        sigma: Callable[[Float32[Array, "3"]], Float32[Array, ""]],
        pose: RadarPose, weight: Float32[Array, ""]
    ) -> Float32[Array, "nr"]:
        """Render a single doppler column for a radar image.

        Parameters
        ----------
        t_sensor: Sensor-space rays on the unit sphere.
        sigma: Field function.
        pose: Sensor pose.
        weight: Sample size weight.

        Returns
        -------
        Rendered column for one doppler value and a stack of range values.
        """
        def render_range(r):
            t_world = self.sensor_to_world(r=r, x=t_sensor, pose=pose)
            sigma_samples = jnp.nan_to_num(vmap(sigma)(t_world.T))
            return jnp.mean(sigma_samples) * 2 * jnp.pi * r * weight / self.n

        return vmap(render_range)(self.r)

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
