"""Ray sampling routines.

Conventions
-----------
- The sensor field of view is centered around +x.
- +y is left of +x.
- +z is straight up.
"""

from jaxtyping import Float32, Bool, Array
from beartype.typing import Union, List

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
        r: Union[Float32[Array, "nr"], List[float]] = None,
        d: Union[Float32[Array, "nd"], List[float]] = None
    ) -> None:
        self.r = jnp.array(r).astype(jnp.float32)
        self.d = jnp.array(d).astype(jnp.float32)
        self.theta_lim = theta_lim
        self.phi_lim = phi_lim
        self.n = n
        self.k = k
        self.bin_width = 2 * jnp.pi / n
        self._extents = [min(self.d), max(self.d), min(self.r), max(self.r)]

    def valid_mask(
        self, d: Float32[Array, ""], pose: RadarPose
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
        x, y, z = project_angle(d, jnp.arange(self.n) * self.bin_width, pose)

        theta = jnp.arcsin(z)
        phi = jnp.arcsin(y * jnp.cos(theta))
        return (
            (theta < self.theta_lim) & (theta > -self.theta_lim)
            & (phi < self.phi_lim) & (phi > -self.phi_lim)
            & (x > 0))

    def sample_rays(
            self, key, d: Float32[Array, ""],
            valid_psi: Bool[Array, "n"], pose: RadarPose
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
