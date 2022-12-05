"""Stochastic integration routines."""

import torch
from sample import Pose, vsample_points


def render(
        sigma, r: torch.Tensor, d: torch.Tensor, pose: Pose,
        n: int = 360, k: int = 120):
    """Render radar image for range x doppler bins.

    Parameters
    ----------
    sigma : function(float[batch, 3]) --> float[batch]
        Field function.
    r : torch.Tensor
        Selected range bins.
    d : torch.Tensor
        Selected doppler bins.
    pose : Pose
        Sensor pose parameters.
    n : int
        Bin angular resolution.
    k : int
        Number of samples for stochastic integration.
    """
    rr, dd = torch.meshgrid(r, d, indexing='ij')
    return stochastic_integration(
        sigma, rr.reshape(-1), dd.reshape(-1), pose, n=n, k=k
    ).reshape(len(r), len(d))


def loss(
        sigma, measured, r: torch.Tensor, d: torch.Tensor, pose: Pose,
        epsilon: float = 0.01, n: int = 360, k: int = 120):
    """Get loss for an input image."""
    pass


def stochastic_integration(
        sigma, r: torch.Tensor, d: torch.Tensor, pose: Pose, n: int = 360, k: int = 120):
    """Stochastic integration over the field sigma.

    Parameters
    ----------
    sigma : function(float[batch, 3]) --> float[batch]
        Field function.
    rd : float[2, batch]
        Radar measurements: ({range, doppler}, batch)
    pose : Pose
        Sensor pose parameters.
    n : int
        Bin angular resolution.
    k : int
        Number of samples for stochastic integration.
    """
    batch = r.shape[0]
    samples, weights = vsample_points(r, d, pose, n=n, k=k)

    s_hat = sigma(samples.reshape(-1, 3)).reshape(batch, k)
    return torch.mean(s_hat, axis=1) * 2 * torch.pi * r * weights / n
