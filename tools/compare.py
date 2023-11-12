"""Create a video comparison of different methods (and ground truth)."""

import os
import math
from tqdm import tqdm
import matplotlib as mpl
from functools import partial

from jax import numpy as jnp
import jax
import numpy as np
import cv2
import h5py

from dart import DartResult


def _parse(p):
    p.add_argument(
        "-p", "--path", nargs='+', required=True,
        help="Files to load and render; are assumed to use the same dataset.")
    p.add_argument(
        "-c", "--fourcc", help="Format fourcc code.", default="mp4v")
    p.add_argument(
        "-o", "--out", default="results/compare.mp4", help="Output file.")
    p.add_argument(
        "-f", "--fps", default=15.625, type=float,
        help="Video framerate; set as 1 / (scan_dt * stride) for 1:1 time.")
    p.add_argument(
        "-s", "--size", type=int, default=128,
        help="Vertical size to rescale each plot to.")
    p.add_argument(
        "-w", "--width", type=int, default=1024,
        help="Horizontal size to rescale to.")
    p.add_argument(
        "-b", "--batch", type=int, default=512, help="Batch size.")
    return p


def _resize(img, size):
    if img.shape[:2] != size:
        return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
    else:
        return img


def _main(args):
    results = [DartResult(p) for p in args.path]

    cmap = jnp.array(mpl.colormaps['viridis'].colors)
    images = [
        h5py.File(results[0].DATASET)["rad"],
        h5py.File(os.path.join(
            os.path.dirname(results[0].DATASET), "simulated.h5"))["rad"]
    ] + [res.open(res.RADAR)["rad"] for res in results]

    validx = np.zeros(images[0].shape[0], dtype=bool)
    valset = np.load(os.path.join(args.path[0], "metadata.npz"))["val"]
    validx[valset] = True

    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    output_dims = (args.width, args.size * (len(images)))
    out = cv2.VideoWriter(args.out, fourcc, args.fps, output_dims)

    _loadrad = jax.jit(partial(results[0].colorize_radar, square=False))

    frame_idx = 0
    for _ in tqdm(range(math.ceil(images[0].shape[0] / args.batch))):
        batch = [np.asarray(_loadrad(cmap, x[:args.batch])) for x in images]
        images = [x[args.batch:] for x in images]

        for frames in zip(*batch):
            resized = [_resize(x, (args.width, args.size)) for x in frames]
            f = np.concatenate(resized, axis=0)

            label = "{}{}".format("v" if validx[frame_idx] else "t", frame_idx)
            cv2.putText(
                f, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2, cv2.LINE_AA)
            frame_idx += 1

            # RGB -> BGR since OpenCV assumes BGR
            out.write(f[:, :, [2, 1, 0]])

    out.release()
