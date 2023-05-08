"""Virtual Camera routines."""

from functools import partial

from jax import numpy as jnp
from jax import vmap

from jaxtyping import Float32, Array
from beartype.typing import NamedTuple

from .pose import sensor_to_world
from . import types


class VirtualCamera(NamedTuple):
    """Depth "camera" model.

    Attributes
    ----------
    w, h: width, height of camera in pixels.
    k: intrinsic matrix.
    d: depth resolution.
    max_depth: maximum depth to render to.
    """

    # w: int
    # h: int
    # k: Float32[Array, "3 3"]
    d: float
    max_depth: float

    def render_pixel(
        self, t: Float32[Array, "3"], sigma: types.SigmaField,
        pose: types.RadarPose
    ) -> tuple[Float32[Array, ""], Float32[Array, ""]]:
        """Render a single pixel.

        Parameters
        ----------
        t: Sensor-space pixel ray on the unit sphere.
        sigma: Field function.
        pose: Sensor pose.

        Returns
        -------
        depth: depth of the point that would contribute the most to a radar
            rendering with range bins `linspace(0, d, max_depth)`.
        sigma: the reflectance value at that point.
        """
        def project(r):
            t_world = sensor_to_world(r=r, t=t.reshape(3, 1), pose=pose)[:, 0]
            return sigma(t_world)

        sigma_samples, alpha_samples = vmap(project)(
            jnp.arange(0, self.max_depth, self.d))

        transmitted = jnp.concatenate([
            jnp.zeros((1)), jnp.cumsum(alpha_samples[:-1])])
        amplitude = sigma_samples  # * jnp.exp(transmitted * 0.1)
        d = jnp.argmax(amplitude)
        return d.astype(float), jnp.max(sigma_samples)

    def render(
        self, sigma: types.SigmaField, pose: types.RadarPose
    ) -> Float32[Array, "w h 2"]:
        """Render single virtual depth camera image.

        Parameters
        ----------
        sigma: Field function.
        pose: Sensor pose.

        Returns
        -------
        Image with (depth, sigma) channels.
        """
        x = jnp.linspace(-1, 1, 64)
        y = jnp.linspace(-1, 1, 64)
        xx, yy = jnp.meshgrid(x, y)
        xyz = jnp.stack([xx, yy, 0.2 * jnp.ones_like(yy)], axis=2)
        xyz = xyz / jnp.linalg.norm(xyz, keepdims=True, axis=2)
        return vmap(vmap(
            partial(self.render_pixel, sigma=sigma, pose=pose)))(xyz)
