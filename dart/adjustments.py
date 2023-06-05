"""Pose adjustment objective."""

from jax import numpy as jnp
import haiku as hk

from jaxtyping import Float32, Array
from beartype.typing import Callable, Optional, Union

from . import types


class Identity(hk.Module):
    """No adjustments."""

    @classmethod
    def from_config(cls):
        """Create identity closure."""
        return lambda: cls()

    def __call__(
        self, pose: Optional[types.RadarPose]
    ) -> Union[types.RadarPose, Float32[Array, ""]]:
        """No adjustment."""
        if pose is None:
            return jnp.array(0.0)
        else:
            return pose


class Position(Identity):
    """Position adjustment parameters."""

    def __init__(self, n: int, k: int, alpha: float = 1.) -> None:
        super().__init__()
        self.n = n
        self.k = k
        self.alpha = alpha

    @classmethod
    def from_config(
        cls, n=10000, k=100, alpha=100.
    ) -> Callable[[], "Position"]:
        """Create NGP haiku closure from config items."""
        def closure():
            return cls(n=n, k=k, alpha=alpha)
        return closure

    def __call__(self, pose: types.RadarPose) -> types.RadarPose:
        """Apply pose adjustments."""
        deltas = hk.get_parameter("delta", shape=(self.k, 3), init=jnp.zeros)

        if pose is None:
            return jnp.sum(jnp.abs(jnp.diff(deltas))) * self.alpha
        else:
            raw = pose.i * (self.k - 1) / (self.n - 1)
            left = jnp.floor(raw).astype(jnp.int32)
            right = jnp.ceil(raw).astype(jnp.int32)

            interp = (
                deltas[left] * (raw - left) + deltas[right] * (right - raw))
            deltas_final = jnp.where(left == right, deltas[left], interp)

            return types.RadarPose(
                v=pose.v, p=pose.p, q=pose.q, s=pose.s,
                x=pose.x + deltas_final, A=pose.A, i=pose.i)
