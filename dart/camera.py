"""Virtual Camera routines."""

from matplotlib import colors
from functools import partial
from scipy.io import loadmat

import numpy as np
from jax import numpy as jnp
from jax import vmap

from jaxtyping import Float32, Array
from beartype.typing import NamedTuple

from .pose import sensor_to_world
from . import types


class VirtualCameraImage(NamedTuple):
    """Rendered virtual camera image.

    Attributes
    ----------
    d: distance to (occlusion-adjusted) brightest point along a ray relative to
        `max_depth`; has range [0, 1].
    sigma: radar return of the brightest point for that ray.
    a: amplitude contribution of that ray.
    """

    d: Float32[Array, "w h"]
    sigma: Float32[Array, "w h"]
    a: Float32[Array, "w h"]

    def to_rgb(
        self, clip: float = 5.0, range: tuple[float, float] = (0, 0.5),
        sat_decay_slope: float = 4.0
    ) -> Float32[Array, "w h 3"]:
        """Convert to RGB image.

        Parameters
        ----------
        clip: Percentile to clip extreme sigma values by.
        range: Hue range to use for different sigma values.
        sat_decay_slope: Slope for the saturation to decay to 0 for low-sigma
            points.

        Returns
        -------
        Image with floating point RGB in [0, 1].
        """
        lower = np.percentile(self.sigma, clip)
        upper = np.percentile(self.sigma, 100 - clip)
        sigma = (np.clip(self.sigma, lower, upper) - lower) / (upper - lower)
        sigma = sigma * (range[1] - range[0]) + range[0]
        hsv = np.stack([
            sigma, np.minimum(1, sigma * sat_decay_slope), 1 - self.d
        ], axis=-1)
        return (colors.hsv_to_rgb(hsv) * 255).astype(np.uint8)

    @classmethod
    def from_file(cls, file: str) -> "VirtualCameraImage":
        """Load from file."""
        data = loadmat(file)
        return cls(d=data["d"], sigma=data["sigma"], a=data["a"])


class VirtualCamera(NamedTuple):
    """Depth "camera" model.

    Attributes
    ----------
    w, h: width, height of camera in pixels.
    k: intrinsic matrix.
    d: depth resolution.
    max_depth: maximum depth to render to.
    f: focal length.
    clip: minimum return threshold.
    """

    # w: int
    # h: int
    # k: Float32[Array, "3 3"]
    d: int
    max_depth: float
    f: float
    clip: float

    def render_pixel(
        self, t: Float32[Array, "3"], pose: types.RadarPose,
        field: types.SigmaField
    ) -> tuple[Float32[Array, ""], Float32[Array, ""], Float32[Array, ""]]:
        """Render a single pixel along `linspace(0, d, max_depth)`.

        Parameters
        ----------
        t: Sensor-space pixel ray on the unit sphere.
        pose: Sensor pose.
        field: Field function.

        Returns
        -------
        (d, sigma, alpha) pixel.
        """
        def project(r):
            t_world = sensor_to_world(r=r, t=t.reshape(3, 1), pose=pose)[:, 0]
            dx = pose.x - t_world
            dx_norm = dx / jnp.linalg.norm(dx)
            return field(t_world, dx=dx_norm)

        sigma, alpha = vmap(project)(jnp.linspace(0, self.max_depth, self.d))

        # transmitted = jnp.concatenate([jnp.zeros((1)), jnp.cumsum(alpha[:-1])])
        amplitude = jnp.nan_to_num(sigma, nan=0.0, copy=False)  # * jnp.exp(transmitted * 0.1)

        d_idx = jnp.argmax(amplitude)
        d_clip = jnp.where(amplitude[d_idx] >= self.clip, d_idx / self.d, 1)

        return d_clip, amplitude[d_idx], jnp.sum(amplitude)

    def render(
        self, pose: types.RadarPose, field: types.SigmaField
    ) -> VirtualCameraImage:
        """Render single virtual depth camera image.

        Notes
        -----
        `z` is "reversed" (positive z has lower index) to follow image
        conventions where the top row has index 0.

        Parameters
        ----------
        pose: Sensor pose.
        field: Sigma/alpha field function.

        Returns
        -------
        Image with (depth, sigma, amplitude) channels.
        """
        y = jnp.linspace(-1, 1, 128)
        z = jnp.linspace(1, -1, 128)
        yy, zz = jnp.meshgrid(y, z)
        xyz = jnp.stack([self.f * jnp.ones_like(yy), yy, zz], axis=2)
        xyz = xyz / jnp.linalg.norm(xyz, keepdims=True, axis=2)
        distance, sigma, amplitude = vmap(vmap(
            partial(self.render_pixel, pose=pose, field=field)))(xyz)
        return VirtualCameraImage(d=distance, sigma=sigma, a=amplitude)
