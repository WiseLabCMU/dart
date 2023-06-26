"""SSIM baseline with synthetic noise."""

import os
import numpy as np
import jax
import h5py
from jax import numpy as jnp
from dart.utils import ssim
from functools import partial

from dart import DartResult


_desc = "Compute SSIM references for synthetic noise."


def _parse(p):
    p.add_argument("-p", "--path", help="Path to result directory.")
    p.add_argument(
        "--clip", type=float, default=99.9,
        help="Clip maximum value by percentile (required for long-tailed "
        "floating point input to SSIM).")
    p.add_argument(
        "--eps", type=float, default=5e-3,
        help="Threshold to exclude empty regions from SSIM calculation.")
    p.add_argument(
        "--size", type=int, default=11, help="Filter size for SSIM.")
    p.add_argument(
        "--sigma", type=float, default=1.5, help="Filter gaussian sigma.")
    p.add_argument(
        "--psnr", type=int, nargs='+', default=[20, 30],
        help="PSNR Values to use for reference.")
    return p


def _main(args):

    print("Loading...")
    result = DartResult(args.path)
    validx = result.load(result.VALSET)["val"]

    def _scale(x):
        pmax = np.percentile(x, args.clip)
        return jnp.clip(x, 0.0, pmax) / pmax

    gt = h5py.File(result.DATASET)['rad'][validx]
    gt_clip = _scale(gt)

    for psnr_target_db in args.psnr:
        sigma = 1 / np.sqrt(10**(psnr_target_db / 10.0))
        noise = np.random.normal(size=gt_clip.shape, scale=sigma)
        pred_clip = np.clip(gt_clip + noise, 0, 1)

        print("Running...")
        ssim_func = jax.jit(jax.vmap(partial(
            ssim, max_val=1.0, eps=args.eps, filter_size=args.size,
            filter_sigma=args.sigma)))

        res, weight = ssim_func(gt_clip, pred_clip)
        print("SSIM @ {}db: {}".format(psnr_target_db, np.mean(res)))

        out = "ssim_{}db.npz".format(psnr_target_db)
        np.savez(os.path.join(args.path, out), ssim=res, weight=weight)
