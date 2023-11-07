"""Pose adjustment objective."""

from jax import numpy as jnp
import haiku as hk

from jaxtyping import Float32, Array
from beartype.typing import Callable, Optional, Literal, overload

from . import types


class Adjustment(hk.Module):
    """Adjustment base class."""

    @classmethod
    def from_config(cls, *args, **kwargs) -> Callable[[], "Adjustment"]:
        """Create identity closure."""
        return lambda: cls(*args, **kwargs)  # type: ignore

    @overload
    def __call__(self, pose: Literal[None]) -> Float32[Array, ""]:
        ...

    @overload
    def __call__(self, pose: types.RadarPose) -> types.RadarPose:
        ...

    def __call__(self, pose: Optional[types.RadarPose]):
        """No adjustment.
        
        Returns the regularization value for the adjustment (0.0) if `pose` is
        `None`; otherwise, returns the adjusted pose.
        """
        if pose is None:
            return jnp.array(0.0)
        else:
            return pose


class Identity(Adjustment):
    """No adjustments."""

    pass


class Position(Adjustment):
    """Position adjustment parameters.
    
    Parameters
    ----------
    n: interval between keypoints.
    k: total number of keypoints.
    alpha: regularization multiplier.
    """

    def __init__(self, n: int, k: int, alpha: float = 1.) -> None:
        super().__init__()
        self.n = n
        self.k = k
        self.alpha = alpha

    def __call__(self, pose: Optional[types.RadarPose]):
        """Apply pose adjustments."""
        delta = hk.get_parameter("delta", shape=(self.k, 3), init=jnp.zeros)

        if pose is None:
            return 0.0  # jnp.sum(jnp.linalg.norm(delta, axis=1)) * self.alpha
        else:
            raw = pose.i.astype(jnp.float32) / self.n
            left = jnp.floor(raw).astype(jnp.int32)
            dx = (
                delta[left] * (left + 1 - raw)
                + delta[left + 1] * (raw - left))

            return types.RadarPose(
                v=pose.v, p=pose.p, q=pose.q, s=pose.s,
                x=pose.x + dx, A=pose.A, i=pose.i)
