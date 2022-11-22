"""Routines to make samples for integration.

Usage
-----
1. Generate pose::

    pose = make_pose(velocity_vector, theta_lim=pi/12, phi_lim=pi/3)

This generates an orthonormal basis which includes the velocity vector. The
velocity is assumed to be in sensor-space (+x is facing forward). Pose
generation is not vectorized (albeit very cheap), and should not be run in jit.

2. Draw samples::

    samples, weights = sample(r_batch, d_batch, pose, n=360, k=120)

``n`` is the number of bins, and ``k`` is the number of samples to pull from
valid bins. Sampling is currently IID. The ``weight`` indicates the number of
valid bins for each sample; to obtain exact units for the stochastic integral::

    stochastic_integral = mean(field(samples)) * 2 * pi * r * weights / n
"""

from collections import namedtuple

import numpy as np
from scipy import linalg

import torch
from functorch import vmap


Pose = namedtuple(
    "Pose", ["v", "s", "p", "q", "x", "A", "theta_lim", "phi_lim"])


def make_pose(
        v: np.ndarray, x: np.array, A: np.ndarray,
        theta_lim: float = np.pi / 12, phi_lim: float = np.pi / 3):
    """Create pose data for a single scan.

    NOTE: Must be run outside of torch.jit due to `scipy.linalg.orth`. The
    result can be safely passed into a jit function.

    Parameters
    ----------
    v : float[3]
        Velocity vector; in global coordinates.
    x : float[3]
        Sensor location; in global coordinates.
    A : float[3, 3]
        3D rotation matrix for the sensor pose; should transform from
        sensor-space to world-space (left 3x3 for affine transform).
    theta_lim : float
        Bounds on elevation angle theta; +/- pi/12 (15 degrees) by default.
    phi_lim : float
        Bounds on azimuth angle phi; +/- pi/3 (60 degrees) by default.

    Returns
    -------
    Pose
        Created pose object.
    """
    # Transform velocity to sensor space and separate magnitude
    v_sensor = np.matmul(np.linalg.inv(A), v)
    s = np.linalg.norm(v_sensor)
    v = v_sensor / s

    # This takes an identity matrix, mods out v, and turns the remainder
    # into an orthonormal basis.
    # Using scipy.linalg.orth lets us use scipy for numerical stability.
    p, q = linalg.orth(np.stack([u - np.dot(u, v) * v for u in np.eye(3)])).T

    # Strange shapes are used later to expand specific dimensions.
    return Pose(
        # Velocity direction and magnitude in sensor space
        v=torch.Tensor(v).reshape(1, 3, 1), s=torch.Tensor(s),
        # Orthonormal basis of integration circle / arc(s)
        p=torch.Tensor(p).reshape(1, 3, 1), q=torch.Tensor(q).reshape(1, 3, 1),
        # Current pose: position (x) and rotation (A)
        x=torch.Tensor(x).reshape(3, 1), A=torch.Tensor(A),
        # Sensor parameters
        theta_lim=theta_lim, phi_lim=phi_lim)


def project(rd: torch.Tensor, psi: torch.Tensor, pose: Pose):
    """Generate projections (vectorized).

    Parameters
    ----------
    rd : float[2, batch]
        Radar measurement: ({radius, doppler}, batch)
    psi : float[batch, k]
        Angles for each (r, d) pair.
    pose : Pose
        (v, s) velocity vector and (p, q) basis.

    Returns
    -------
    float[3, batch, k]
        xyz output coords.
    """
    batch = rd.shape[0]
    psi = psi.reshape(batch, 1, -1)
    r_prime = torch.sqrt(1 - (rd[:, 0] / pose.s)**2).reshape(batch, 1, 1)

    # psi is (batch, 1, k)
    # p, q are (1, 3, 1)
    # r_prime, d are (batch, 1, 1)
    return rd[:, 0].reshape(batch, 1, 1) * (
        r_prime * (pose.p * torch.cos(psi) + pose.q * torch.sin(psi))
        + pose.v * rd[:, 1].reshape(batch, 1, 1) / pose.s)


def valid_mask(
        rd: torch.Tensor, psi: torch.Tensor, pose: Pose):
    """Get valid psi values as a mask.

    Parameters
    ----------
    rd : float[2, batch]
        Radar measurement: ({radius, doppler}, batch)
    psi : float[batch, k]
        Angles for each (r, d) pair.
    pose : Pose
        (v, s) velocity vector and (p, q) basis.

    Returns
    -------
    bool[batch, n]
        Output mask for each (batch, bin).
    """
    # swap (batch, 3, n) to (3, batch, n)
    x, y, z = torch.swapaxes(project(rd, psi, pose), 0, 1)

    theta = torch.arcsin(z / rd[:, 0].reshape(-1, 1))
    phi = torch.arcsin(y / (rd[:, 0].reshape(-1, 1) * torch.cos(theta)))
    return (
        (theta < pose.theta_lim) & (theta > -pose.theta_lim)
        & (phi < pose.phi_lim) & (phi > -pose.phi_lim)
        & (x > 0))


def sample_points(rd: torch.Tensor, pose: Pose, n: int = 360, k: int = 120):
    """Sample points from projection.

    Points are sampled IID from the valid bins (for now).

    Parameters
    ----------
    rd : float[batch, 2]
        Radar measurement: (batch, {radius, doppler})
    pose : Pose
        Scan pose parameters.
    n : int
        Number of bins (per full circle)
    k : int
        Number of samples (split across valid bins)

    Returns
    -------
    (float[batch, k, 3], int[batch])
        [0] Sampled points in xyz sensor-space.
        [1] Number of bins for each sample; use as the weight.
    """
    batch = rd.shape[0]
    psi = (torch.arange(n) / n * 2 * torch.pi).expand(batch, -1)
    valid = valid_mask(rd, psi, pose).type(torch.float32)

    bin_width = 2 * np.pi / n
    offsets = (torch.rand((batch, k)) - 0.5) * bin_width
    samples = torch.multinomial(valid, k, replacement=True) / n * 2 * torch.pi
    psi_actual = samples + offsets

    # Swap to xyz space and reshape as (3, batch, k)
    points_sensor = torch.swapaxes(
        project(rd, psi_actual, pose), 0, 1).reshape(3, batch * k)
    # Transformation
    _world = pose.x + torch.matmul(pose.A, points_sensor)
    # Swap back to (batch, k, 3).
    points_world = torch.swapaxes(_world.reshape(3, batch, k), 0, 1)

    return points_world, torch.sum(valid, axis=0)
