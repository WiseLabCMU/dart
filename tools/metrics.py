"""Compute validation-set SSIM for range-doppler-azimuth images."""

import numpy as np
import jax
from tqdm import tqdm
from jax import numpy as jnp
from functools import partial

from ._result import DartResult
from dart import metrics, utils


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
    p.add_argument("--batch", default=1024, type=int, help="Batch size.")

    return p


def _get_val_idx(result):
    data = result["data/data.h5"]
    splits = result["result/metadata.npz"]

    # Remove the first frame since it can have some test columns in it.
    return np.unique(data["frame_idx"][splits["val"]])[1:]  # type: ignore


def _metrics(y_true, y_hat, **kwargs):
    y_true = y_true.astype(jnp.float32)
    y_hat = y_hat.astype(jnp.float32)
    mse, alpha = metrics.mse(y_true, y_hat)
    ssim, weight = metrics.ssim(y_true, y_hat * alpha, **kwargs)
    return {"mse": mse, "alpha": alpha, "ssim": ssim, "weight": weight}


def _main(args):

    result = DartResult(args.path)

    val_idx = _get_val_idx(result)
    mask = result["data/trajectory.h5"]["mask"]
    gt = jnp.array(result["data/radar.h5"]["rad"][mask][val_idx])

    lower, upper = jnp.percentile(gt, np.array([5, 99.9]))
    gt = jnp.clip(gt, lower, upper) / (upper - lower)

    if args.baseline:
        pred = jnp.array(result["data/simulated.h5"]["rad"][val_idx])
        # Clip baseline only separately
        bmax = jnp.max(pred)
        pred = jnp.clip(pred, 0.0, bmax) / bmax
    # elif args.psnr is not None:
    #     sigma = 1 / np.sqrt(10**(args.psnr / 10.0))
    #     noise = np.random.normal(size=gt.shape, scale=sigma)
    #     pred = np.clip(gt + noise, 0, 1)
    else:
        pred = jnp.array(result["result/rad.h5"]["rad"][val_idx])
        pred = jnp.clip(pred, lower, upper) / (upper - lower)

    eval_func = jax.jit(jax.vmap(partial(
        _metrics, max_val=1.0, eps=args.eps, filter_size=args.size,
        filter_sigma=args.sigma)))

    res = []
    for _ in tqdm(range(int(np.ceil(gt.shape[0] / args.batch)))):
        y_true = gt[:args.batch]
        y_hat = pred[:args.batch]
        res.append(eval_func(y_true, y_hat))
        gt = gt[args.batch:]
        pred = pred[args.batch:]

    res = utils.tree_concatenate(res)
    for k in ["alpha", "ssim", "mse"]:
        print("{}: mean={}, median={}".format(
            k, np.nanmean(res[k]), np.nanmedian(res[k])))

    if args.baseline:
        out = result.path("data/metrics_simulated.npz")
    # elif args.psnr is not None:
    #     out = os.path.join(result.DATADIR, "ssim_{}db.npz".format(args.psnr))
    else:
        out = result.path("result/metrics.npz")
    np.savez(out, **res)
