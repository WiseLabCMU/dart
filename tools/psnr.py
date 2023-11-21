"""Calculate reference PSNR."""

import numpy as np
import jax
from jax import numpy as jnp
from jax import random
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
    p.add_argument("--batch", default=256, type=int, help="Batch size.")
    p.add_argument(
        "--psnr", default=[25., 30., 35.], nargs='+', type=float,
        help="Reference PSNR values.")
    p.add_argument("--key", default=42, type=int, help="Random key.")
    return p


def _get_idx(result, split="val"):
    data = result["data/data.h5"]
    splits = result["result/metadata.npz"]
    # Remove the first frame since it can have some test columns in it.
    return np.unique(data["frame_idx"][splits[split]])[1:]  # type: ignore


def _metrics(y, noise_sigmas, lower=0.0, upper=0.0, **kwargs):
    (key, y_true) = y
    y_true = y_true.astype(jnp.float32)
    y_true_clip = (jnp.clip(y_true, lower, upper) - lower) / (upper - lower)
    noise = random.normal(key, y_true.shape)

    def _inner(sigma):
        y_hat_clip = jnp.clip(y_true_clip + noise * sigma, 0, 1)
        ssim, weight = metrics.ssim(y_true_clip, y_hat_clip, **kwargs)
        return {"ssim": ssim, "weight": weight}

    return jax.vmap(_inner)(noise_sigmas)


def _main(args):

    result = DartResult(args.path)

    val_idx = _get_idx(result, split="val")
    mask = result["data/trajectory.h5"]["mask"]
    gt = jnp.array(result["data/radar.h5"]["rad"][mask][val_idx])

    noise_sigmas = 1 / np.sqrt(10**(np.array(args.psnr) / 10.0))

    lower, upper = jnp.percentile(gt, np.array([0.01, 99.99]))
    eval_func = jax.jit(jax.vmap(partial(
        _metrics, max_val=1.0, noise_sigmas=noise_sigmas,
        eps=args.eps, filter_size=args.size,
        filter_sigma=args.sigma, lower=lower, upper=upper)))
    keys = jnp.array(random.split(random.PRNGKey(args.key), gt.shape[0]))
    res = utils.vmap_batch(eval_func, (keys, gt), batch=args.batch)

    for i, p in enumerate(args.psnr):
        print("{}db psnr = SSIM {}".format(p, np.nanmean(res["ssim"][:, i])))
    np.savez(result.path("data/baselines/reference.npz"), **res)
