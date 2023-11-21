"""Compute validation-set SSIM for range-doppler-azimuth images."""

import numpy as np
import jax
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
        "-m", "--mode", default=None,
        help="Baseline mode if passed (lidar, cfar, nearest, psnr)")
    p.add_argument("--batch", default=1024, type=int, help="Batch size.")

    return p


def _get_idx(result, split="val"):
    data = result["data/data.h5"]
    splits = result["result/metadata.npz"]
    # Remove the first frame since it can have some test columns in it.
    return np.unique(data["frame_idx"][splits[split]])[1:]  # type: ignore


def _nearest(result):
    mask = result["data/trajectory.h5"]["mask"]
    train_idx = _get_idx(result, split="train")
    test_idx = _get_idx(result, split="val")
    traj = result["data/trajectory.h5"]

    train_pose = jnp.concatenate(
        [traj["pos"][train_idx], traj["vel"][train_idx]], axis=1)
    test_pose = jnp.concatenate(
        [traj["pos"][test_idx], traj["vel"][test_idx]], axis=1)

    distances = jnp.sum(jnp.square(
        train_pose[None, :, :] - test_pose[:, None, :]), axis=-1)
    indices = jnp.argmin(distances, axis=1)

    train_rad = jnp.array(result["data/radar.h5"]["rad"][mask][train_idx])
    return train_rad[indices]


def _metrics(y, lower=0.0, upper=0.0, **kwargs):
    y_true, y_hat = y
    y_true = y_true.astype(jnp.float32)
    y_hat = y_hat.astype(jnp.float32)
    mse, xi = metrics.mse(y_true, y_hat)

    y_true_clip = (jnp.clip(y_true, lower, upper) - lower) / (upper - lower)
    y_hat_clip = (jnp.clip(y_hat * xi, lower, upper) - lower) / (upper - lower)

    ssim, weight = metrics.ssim(y_true_clip, y_hat_clip, **kwargs)
    return {"mse": mse, "xi": xi, "ssim": ssim, "weight": weight}


def _main(args):

    result = DartResult(args.path)

    val_idx = _get_idx(result, split="val")
    mask = result["data/trajectory.h5"]["mask"]
    gt = jnp.array(result["data/radar.h5"]["rad"][mask][val_idx])

    lower, upper = jnp.percentile(gt, np.array([0.01, 99.99]))

    if args.mode is None:
        out = result.path("result/metrics.npz")
        pred = jnp.array(result["result/rad.h5"]["rad"][val_idx])
    elif args.mode == "lidar":
        out = result.path("data/baselines/lidar.npz")
        pred = jnp.array(result["data/baselines/lidar.h5"]["rad"][val_idx])
    elif args.mode.startswith("cfar"):
        out = result.path("data/baselines/{}.npz".format(args.mode))
        pred = jnp.array(
            result["data/baselines/{}.h5".format(args.mode)]["rad"][val_idx])
    # elif args.mode.startswith("psnr"):
    #     out = result.path("data/baselines/{}.npz".format(args.mode))
    #     noise = 1 / np.sqrt(10**(int(args.psnr.replace("psnr", "")) / 10.0))
    #     pred = gt
    elif args.mode == "nearest":
        out = result.path("data/baselines/nearest.npz")
        pred = _nearest(result)
    else:
        raise ValueError("Unknown mode: {}".format(args.mode))

    eval_func = jax.jit(jax.vmap(partial(
        _metrics, max_val=1.0, eps=args.eps, filter_size=args.size,
        filter_sigma=args.sigma, lower=lower, upper=upper)))
    res = utils.vmap_batch(eval_func, (gt, pred), batch=args.batch)

    for k in ["xi", "ssim", "mse"]:
        print("{}: mean={}, median={}".format(
            k, np.nanmean(res[k]), np.nanmedian(res[k])))
        
    np.savez(out, **res)
