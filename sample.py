"""Routines to make samples for integration.

Usage
-----
1. Generate pose::

    pose = make_pose(velocity_vector, theta_lim=pi/12, phi_lim=pi/3)

This generates an orthonormal basis which includes the velocity vector. The
velocity is assumed to be in sensor-space (+x is facing forward). Pose
generation is not vectorized (albeit very cheap), and should not be run in jit.

2. Draw samples::

    samples, weight = sample(r_batch, d_batch, pose, n=360, k=120)

``n`` is the number of bins, and ``k`` is the number of samples to pull from
valid bins. Sampling is currently IID. The ``weight`` indicates the number of
valid bins for each sample; to obtain exact units for the stochastic integral::

    stochastic_integral = mean(...) * 2 * pi * r * weight / n
"""

from collections import namedtuple

import numpy as np
from scipy import linalg

import torch


Pose = namedtuple("Pose", ["v", "p", "q", "s", "theta_lim", "phi_lim"])


def make_pose(
        v: np.ndarray,
        theta_lim: float = np.pi / 12, phi_lim: float = np.pi / 3):
    """Create pose data for a single scan.

    NOTE: Must be run outside of torch.jit due to `scipy.linalg.orth`.

    Parameters
    ----------
    v : float[3]
        Velocity vector. Assumes radar is facing (1, 0, 0).
    theta_lim : float
        Bounds on elevation angle theta; +/- pi/12 (15 degrees) by default.
    phi_lim : float
        Bounds on azimuth angle phi; +/- pi/3 (60 degrees) by default.

    Returns
    -------
    Pose
        Created pose object.
    """
    s = np.linalg.norm(v)

    # This takes an identity matrix, mods out v, and turns the remainder
    # into an orthonormal basis.
    # Using scipy.linalg.orth lets us use scipy for numerical stability.
    p, q = linalg.orth(
        np.stack([u - np.dot(u, v) / s**2 * v for u in np.eye(3)])).T

    return Pose(
        v=torch.Tensor(v).reshape(1, 3, 1) / s,
        p=torch.Tensor(p).reshape(1, 3, 1),
        q=torch.Tensor(q).reshape(1, 3, 1),
        s=s, theta_lim=theta_lim, phi_lim=phi_lim)


def project(
        r: torch.Tensor, d: torch.Tensor, psi: torch.Tensor, pose: Pose):
    """Generate projections (vectorized).

    Parameters
    ----------
    r : float[batch]
        Radius values.
    d : float[batch]
        Doppler values.
    psi : float[batch, k]
        Angles for each (r, d) pair.
    pose : Pose
        (v, s) velocity vector and (p, q) basis.

    Returns
    -------
    float[3, batch, k]
        xyz output coords.
    """
    batch = r.shape[0]
    psi = psi.reshape(batch, 1, -1)
    r_prime = torch.sqrt(1 - (d / pose.s)**2).reshape(batch, 1, 1)

    # psi is (batch, 1, k)
    # p, q are (1, 3, 1)
    # r_prime, d are (batch, 1, 1)
    return r.reshape(batch, 1, 1) * (
        r_prime * (pose.p * torch.cos(psi) + pose.q * torch.sin(psi))
        + pose.v * d.reshape(batch, 1, 1) / pose.s)


def valid_mask(
        r: torch.Tensor, d: torch.Tensor, psi: torch.Tensor, pose: Pose):
    """Get valid psi values as a mask.

    Parameters
    ----------
    r : float[batch]
        Radius values.
    d : float[batch]
        Doppler values.
    psi : float[batch, k]
        Angles for each (r, d) pair.
    pose : Pose
        (v, s) velocity vector and (p, q) basis.

    Returns
    -------
    bool[batch, k]
        Output mask for each (batch, bin).
    """
    # swap (batch, 3, k) to (3, batch, k)
    x, y, z = torch.swapaxes(project(r, d, psi, pose), 0, 1)

    theta = torch.arcsin(z / r.reshape(-1, 1))
    phi = torch.arcsin(y / (r.reshape(-1, 1) * torch.cos(theta)))
    return (
        (theta < pose.theta_lim) & (theta > -pose.theta_lim)
        & (phi < pose.phi_lim) & (phi > -pose.phi_lim)
        & (x > 0))


def sample(
        r: torch.Tensor, d: torch.Tensor, pose: Pose,
        n: int = 360, k: int = 120):
    """Sample points from projection.

    Points are sampled IID from the valid bins (for now).

    Parameters
    ----------
    r : float[batch]
        Radius values.
    d : float[batch]
        Doppler values.
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
    batch = r.shape[0]
    psi = (torch.arange(n) / n * 2 * torch.pi).expand(batch, -1)
    valid = valid_mask(r, d, psi, pose).type(torch.float32)

    bin_width = 2 * np.pi / n
    offsets = (torch.rand((batch, k)) - 0.5) * bin_width
    samples = (
        torch.multinomial(valid, k, replacement=True) / n * 2 * torch.pi)
    psi_actual = samples + offsets

    return project(r, d, psi_actual, pose), torch.sum(valid, axis=0)
