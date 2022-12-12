"""Ray sampling routines.

Conventions
-----------
- The sensor field of view is centered around +x.
- +y is left of +x.
- +z is straight up.
"""

from jaxtyping import Float32, Bool, Array
from beartype.typing import Optional

from jax import numpy as jnp
from jax import random

from .pose import RadarPose, project_angle
from .sensor_utils import VirtualRadarUtilMixin
from .sensor_column import VirtualRadarColumnMixins


class VirtualRadar(VirtualRadarUtilMixin, VirtualRadarColumnMixins):
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
        n: int = 256, k: int = 128,
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
        x, y, z = project_angle(d, psi, pose)

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
        points = project_angle(d, psi_actual, pose)
        return points
