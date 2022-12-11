"""Sensor pose utilities."""

from jaxtyping import Float32, Array
from beartype.typing import NamedTuple

from jax import numpy as jnp


class RadarPose(NamedTuple):
    """Radar pose parameters (24 x 4 = 96 bytes).

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


def sensor_to_world(
    r: Float32[Array, ""], t: Float32[Array, "3 k"], pose: RadarPose
) -> Float32[Array, "3 k"]:
    """Project points to world-space.

    Parameters
    ----------
    r: Range bin.
    t: Positions in unit-sphere sensor-space.
    pose: Sensor pose.

    Returns
    -------
    Projected points at the specified range in world-space.
    """
    return pose.x.reshape(3, 1) + jnp.matmul(pose.A, r * t)


def project_angle(
    d: Float32[Array, ""], psi: Float32[Array, "n"], pose: RadarPose
) -> Float32[Array, "3 n"]:
    """Project angles to intersection circle on a unit sphere.

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
