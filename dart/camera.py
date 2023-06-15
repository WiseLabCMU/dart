"""Virtual Camera routines."""

from functools import partial

from jax import numpy as jnp
from jax import vmap

from jaxtyping import Float, Float32, Float16, Array, UInt8
from beartype.typing import NamedTuple

from .dataset import load_arrays
from .pose import sensor_to_world
from . import types
from .jaxcolors import hsv_to_rgb


class VirtualCameraImage(NamedTuple):
    """Rendered virtual camera image.

    Attributes
    ----------
    d: distance to (occlusion-adjusted) brightest point along a ray relative to
        `max_depth`; has range [0, 1].
    sigma: radar return of the brightest point for that ray.
    a: amplitude contribution of that ray.
    """

    d: Float[Array, "... w h"]
    sigma: Float[Array, "... w h"]
    a: Float[Array, "... w h"]

    def to_rgb(
        self, clip: float = 5.0, range: tuple[float, float] = (0, 0.5),
        sat_decay_slope: float = 4.0
    ) -> UInt8[types.ArrayLike, "... 3"]:
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
        upper = jnp.percentile(self.sigma, 100 - clip)
        sigma = jnp.clip(self.sigma, 0.0, upper) / upper
        sigma = sigma * (range[1] - range[0]) + range[0]
        hsv = jnp.stack([
            sigma, jnp.minimum(1, sigma * sat_decay_slope), 1 - self.d
        ], axis=-1)
        return (hsv_to_rgb(hsv) * 255).astype(jnp.uint8)

    @classmethod
    def from_file(cls, file: str) -> "VirtualCameraImage":
        """Load from file."""
        data = load_arrays(file)
        return cls(d=data["d"], sigma=data["sigma"], a=data["a"])


class VirtualCamera(NamedTuple):
    """Depth "camera" model.

    Attributes
    ----------
    d: depth resolution.
    max_depth: maximum depth to render to.
    f: focal length (perfect camera with fx = fy).
    size: sensor size (dx, dy) as an offset to the center; always symmetric.
    res: image resolution (width, height).
    clip: minimum return threshold.
    """

    d: int
    max_depth: float
    f: float
    size: tuple[float, float]
    res: tuple[int, int]
    clip: float

    def render_pixel(
        self, t: Float32[Array, "3"], pose: types.RadarPose,
        field: types.SigmaField
    ) -> tuple[Float16[Array, ""], Float16[Array, ""], Float16[Array, ""]]:
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
        # Direction is the same for all ranges.
        dx = jnp.matmul(pose.A, t)

        def project(r):
            t_world = sensor_to_world(r=r, t=t.reshape(3, 1), pose=pose)[:, 0]
            return field(t_world, dx=dx)

        sigma, alpha, _ = vmap(project)(
            jnp.linspace(0, self.max_depth, self.d))

        # tx = jnp.concatenate([jnp.zeros((1)), jnp.cumsum(alpha[:-1])])
        rx = jnp.nan_to_num(sigma, nan=0.0, copy=False).astype(jnp.float16)

        d_idx = jnp.argmax(rx)
        d_clip = jnp.where(rx[d_idx] >= self.clip, d_idx / self.d, 1.0)

        return d_clip.astype(jnp.float16), rx[d_idx], jnp.sum(rx)

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
        y = jnp.linspace(-self.size[0], self.size[0], self.res[0])
        z = jnp.linspace(self.size[1], -self.size[1], self.res[1])
        yy, zz = jnp.meshgrid(y, z)
        xyz = jnp.stack([self.f * jnp.ones_like(yy), yy, zz], axis=2)
        xyz = xyz / jnp.linalg.norm(xyz, keepdims=True, axis=2)
        distance, sigma, amplitude = vmap(vmap(
            partial(self.render_pixel, pose=pose, field=field)))(xyz)
        return VirtualCameraImage(d=distance, sigma=sigma, a=amplitude)
