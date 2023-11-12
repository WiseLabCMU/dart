"""Compute validation-set SSIM for range-doppler-azimuth images."""

import os
import numpy as np
import jax
import h5py
from jax import numpy as jnp
from dart.utils import ssim
from functools import partial

from dart import DartResult


def _parse(p):
    p.add_argument("-p", "--path", help="Path to result directory.")
    p.add_argument(
        "--eps", type=float, default=5e-3,
        help="Threshold to exclude empty regions from SSIM calculation.")
    p.add_argument(
        "--size", type=int, default=11, help="Filter size for SSIM.")
    p.add_argument(
        "--sigma", type=float, default=1.5, help="Filter gaussian sigma.")
    p.add_argument(
        "--baseline", default=False, action='store_true',
        help="Run baseline SSIM on the validation set for this method.")
    p.add_argument(
        "--psnr", default=None, type=int,
        help="If passed, run PSNR reference instead.")

    return p


def _main(args):

    print("Loading...")
    result = DartResult(args.path)
    validx = np.sort(result.load(result.VALSET)["val"])

    gt = np.array(h5py.File(result.DATASET)['rad'])[validx]
    pmax = np.max(gt)
    gt = jnp.clip(gt, 0.0, pmax) / pmax

    if args.baseline:
        pred = np.array(h5py.File(
            os.path.join(result.DATADIR, "simulated.h5"))["rad"])[validx]

        # Clip baseline only separately
        bmax = np.max(pred)
        pred = jnp.clip(pred, 0.0, bmax) / bmax
    elif args.psnr is not None:
        sigma = 1 / np.sqrt(10**(args.psnr / 10.0))
        noise = np.random.normal(size=gt.shape, scale=sigma)
        pred = np.clip(gt + noise, 0, 1)
    else:
        pred = result.open(result.RADAR)['rad'][validx]
        pred = jnp.clip(pred, 0.0, pmax) / pmax

    print("Running...")
    ssim_func = jax.jit(jax.vmap(partial(
        ssim, max_val=1.0, eps=args.eps, filter_size=args.size,
        filter_sigma=args.sigma)))

    res, weight = ssim_func(gt, pred)
    print("SSIM: mean={}, median={}".format(
        jnp.nanmean(res), jnp.nanmedian(res)))

    if args.baseline:
        out = os.path.join(result.DATADIR, "ssim_simulated.npz")
    elif args.psnr is not None:
        out = os.path.join(result.DATADIR, "ssim_{}db.npz".format(args.psnr))
    else:
        out = os.path.join(args.path, "ssim.npz")

    np.savez(out, ssim=res, weight=weight)
