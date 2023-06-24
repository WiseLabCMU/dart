"""SSIM evaluation metric."""

import os
import numpy as np
import jax
import h5py
from jax import numpy as jnp
from dart.utils import ssim
from functools import partial

from dart import DartResult


_desc = "Compute SSIM for range-doppler-azimuth images."


def _parse(p):
    p.add_argument("-p", "--path", help="Path to result directory.")
    p.add_argument(
        "--clip", type=float, default=99.9,
        help="Clip maximum value by percentile (required for long-tailed "
        "floating point input to SSIM).")
    p.add_argument(
        "--eps", type=float, default=5e-3,
        help="Threshold to exclude empty regions from SSIM calculation.")
    return p


def _main(args):

    result = DartResult(args.path)
    validx = result.load(result.VALSET)["val"]

    gt = h5py.File(result.DATASET)['rad'][validx]
    pred = result.open(result.RADAR)['rad'][validx]

    ssim_func = jax.jit(jax.vmap(partial(ssim, max_val=1.0, eps=args.eps)))
    pmax = np.percentile(gt, args.clip, axis=(1, 2, 3))[:, None, None, None]

    gt_clip = jnp.clip(gt, 0.0, pmax) / pmax
    pred_clip = jnp.clip(pred, 0.0, pmax) / pmax

    res = ssim_func(gt_clip, pred_clip)
    print("SSIM:", np.mean(res))
    np.savez(os.path.join(args.path, "ssim.npz"), ssim=res)
