"""Routines to make samples for integration.

Usage
-----
1. Generate pose::

    pose = make_pose(velocity_vector, theta_lim=pi/12, phi_lim=pi/3)

This generates an orthonormal basis which includes the velocity vector. The
velocity is assumed to be in sensor-space (+x is facing forward). Pose
generation is not vectorized (albeit very cheap), and should not be run in jit.

2. Draw samples::

    samples, weights = sample_points(r, d, pose, n=360, k=120)

``n`` is the number of bins, and ``k`` is the number of samples to pull from
valid bins. Sampling is currently IID. The ``weight`` indicates the number of
valid bins for each sample; to obtain exact units for the stochastic integral::

    stochastic_integral = mean(field(samples)) * 2 * pi * r * weights / n
"""

from collections import namedtuple

import numpy as np

import torch
from functorch import vmap


Pose = namedtuple(
    "Pose", ["v", "s", "p", "q", "x", "A", "theta_lim", "phi_lim"])


def make_pose(v, x, A, theta_lim=torch.pi / 12, phi_lim=torch.pi / 3):
    """Create pose data for a single scan.

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
    v_sensor = torch.matmul(torch.linalg.inv(A), v)
    s = torch.linalg.norm(v_sensor)
    v = v_sensor / s

    # This takes an identity matrix, mods out v, and turns the remainder
    # into an orthonormal basis using SVD for best stability.
    _, _, _V = torch.linalg.svd(torch.eye(3) - torch.outer(v, v))
    p, q = _V[:2]

    return Pose(
        v=v, s=s, p=p, q=q, x=x, A=A,
        theta_lim=torch.tensor(theta_lim), phi_lim=torch.tensor(phi_lim))


def project(r: torch.Tensor, d: torch.Tensor, psi: torch.Tensor, pose: Pose):
    """Generate projections.

    Parameters
    ----------
    r : float
        Range bin.
    d : float
        Doppler bin.
    psi : float[k]
        Angles for the (r, d) pair.
    pose : Pose
        (v, s) velocity vector and (p, q) basis.

    Returns
    -------
    float[3, k]
        xyz output coords.
    """
    psi = psi.reshape(1, -1)
    r_prime = torch.sqrt(1 - (r / pose.s)**2)

    # psi is (1, k); p, q are (3, 1)
    return r * (
        r_prime * (
            pose.p.reshape(3, 1) * torch.cos(psi)
            + pose.q.reshape(3, 1) * torch.sin(psi))
        + pose.v.reshape(3, 1) * d / pose.s)


def valid_mask(
        r: torch.Tensor, d: torch.Tensor, psi: torch.Tensor, pose: Pose):
    """Get valid psi values as a mask.

    Returns
    -------
    bool[n]
        Output mask for each bin.
    """
    x, y, z = project(r, d, psi, pose)

    theta = torch.arcsin(z / r)
    phi = torch.arcsin(y / r * torch.cos(theta))
    return (
        (theta < pose.theta_lim) & (theta > -pose.theta_lim)
        & (phi < pose.phi_lim) & (phi > -pose.phi_lim)
        & (x > 0))


def sample_points(
        r: torch.Tensor, d: torch.Tensor,
        pose: Pose, n: int = 360, k: int = 120):
    """Sample points from projection.

    Points are sampled IID from the valid bins (for now).

    Parameters
    ----------
    r : float
        Range bin.
    d : float
        Doppler bin.
    pose : Pose
        Scan pose parameters.
    n : int
        Number of bins (per full circle)
    k : int
        Number of samples (split across valid bins)

    Returns
    -------
    (float[3, k], int)
        [0] Sampled points in xyz sensor-space.
        [1] Number of bins; use as the weight.
    """
    psi = torch.arange(n) / n * 2 * torch.pi
    valid = valid_mask(r, d, psi, pose).type(torch.float32)

    bin_width = 2 * np.pi / n
    offsets = (torch.rand(k) - 0.5) * bin_width
    samples = torch.multinomial(
        valid + 1e-10, k, replacement=True) / n * 2 * torch.pi
    psi_actual = samples + offsets

    points_sensor = project(r, d, psi_actual, pose)
    points_world = pose.x.reshape(3, 1) + torch.matmul(pose.A, points_sensor)
    return points_world, torch.sum(valid)


def _sample_points(
        r: torch.Tensor, d: torch.Tensor,
        pose: Pose = None, n: int = 360, k: int = 120):
    return sample_points(r, d, pose, n, k)


def vpsample_points(
        r: torch.Tensor, d: torch.Tensor,
        pose: Pose, n: int = 360, k: int = 120):
    """Vectorized over range, doppler, and pose.

    NOTE: Pytorch seems to have a bug related to namedtuple types that prevents
    this from being jit compiled.
    """
    return vmap(sample_points, randomness="different")(r, d, pose, n=n, k=k)


def vsample_points(
        r: torch.Tensor, d: torch.Tensor,
        pose: Pose, n: int = 360, k: int = 120):
    """Vectorized over range, doppler, but not pose."""
    return vmap(
        _sample_points, randomness="different")(r, d, pose=pose, n=n, k=k)
