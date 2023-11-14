"""Jax jit+vmap-friendly metrics."""

import jax
from jax import numpy as jnp
import jax.scipy as jsp

from jaxtyping import Float, Array


def mse(
    y_true: Float[Array, "..."], y_hat: Float[Array, "..."]
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Compute MSE with optimal scaling computation.
    
    Parameters
    ----------
    y_true, y_hat: input actual/predicted data.

    Returns
    -------
    mse: MSE of the optimally-scaled `y_hat`.
    alpha: Optimal scale factor.
    """
    alpha = jnp.sum(y_true * y_hat) / jnp.sum(y_hat**2)
    mse = jnp.sum(jnp.square(y_true - alpha * y_hat))
    return mse, alpha


def ssim(
    img0: Float[Array, "width height channels"],
    img1: Float[Array, "width height channels"],
    max_val: float = 1.0, filter_size: int = 11, filter_sigma: float = 1.5,
    k1: float = 0.01, k2: float = 0.03, eps: float = 1e-2
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Jax jit+vmap-friendly SSIM computation from two images.

    The SSIM calculation has been modified to exclude regions which are
    approximately zero in img0 (assumed to be the ground truth) from the SSIM
    calculation.

    Parameters
    ----------
    img0, img1: input images; img0 is considered the 'ground truth'.
    max_val: maximum magnitude that `img0` or `img1` can have.
    filter_size: window size (>1).
    filter_sigma: Bandwidth of the gaussian used for filtering (>0).
    k1, k2: SSIM dampening parameters.
    eps: near-zero exclusion tolerance.

    Returns
    -------
    Mean SSIM of the image.

    References
    ----------
    https://github.com/google/mipnerf/blob/main/internal/math.py
    https://en.wikipedia.org/wiki/Structural_similarity
    """
    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((jnp.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = jnp.exp(-0.5 * f_i)
    filt /= jnp.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return jsp.signal.convolve2d(
            z, f, mode='valid', precision=jax.lax.Precision.HIGHEST)

    filt_fn1 = lambda z: convolve2d(z, filt[:, None])
    filt_fn2 = lambda z: convolve2d(z, filt[None, :])

    # Vmap the blurs to the tensor size, and then compose them.
    num_dims = len(img0.shape)
    map_axes = tuple(list(range(num_dims - 3)) + [num_dims - 1])
    for d in map_axes:
        filt_fn1 = jax.vmap(filt_fn1, in_axes=d, out_axes=d)
        filt_fn2 = jax.vmap(filt_fn2, in_axes=d, out_axes=d)
    filt_fn = lambda z: filt_fn1(filt_fn2(z))

    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = jnp.maximum(0., sigma00)
    sigma11 = jnp.maximum(0., sigma11)
    sigma01 = jnp.sign(sigma01) * jnp.minimum(
        jnp.sqrt(sigma00 * sigma11), jnp.abs(sigma01))

    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom

    mask = (mu0 > eps)
    ssim = jnp.sum(ssim_map * mask) / jnp.sum(mask)
    return ssim, jnp.sum(mask).astype(jnp.float32)
