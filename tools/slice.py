"""Create MRI-style video from map."""

import os
from tqdm import tqdm
import matplotlib as mpl

from dart.dataset import load_arrays
import numpy as np
import cv2


_desc = "Render MRI-style slices, and write each slice to a frame in a video."


def _parse(p):
    p.add_argument("-p", "--path", help="File to load and render.")
    p.add_argument(
        "-c", "--fourcc", help="Format fourcc code.", default="mp4v")
    p.add_argument("-o", "--out", help="Output file.")
    p.add_argument(
        "-f", "--fps", default=10.0, type=float, help="Video framerate.")
    return p


def _colorize(x, k=20):
    conved = sum(x[:, :, i:x.shape[2] - (k - i)] for i in range(k + 1))
    lower = np.percentile(conved, 1)
    upper = np.percentile(conved, 99)
    clipped = np.clip(conved, lower, upper)
    scaled = (clipped - lower) / (upper - lower)
    return (
        mpl.colormaps['viridis'](scaled)[:, :, :, :3] * 255
    ).astype(np.uint8)


def _loadmap(path):
    data = load_arrays(path)
    return _colorize(data["sigma"]), _colorize(data["alpha"])


def _main(args):
    if args.out is None:
        args.out = os.path.join(
            args.path, "slice.mp4")

    sigma, alpha = _loadmap(os.path.join(args.path, "map.mat"))

    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    out = cv2.VideoWriter(
        args.out, fourcc, args.fps, (sigma.shape[1] * 2, sigma.shape[0]))

    for i in tqdm(range(sigma.shape[2])):
        fs = sigma[:, :, i, :]
        fa = alpha[:, :, i, :]
        f = np.flip(np.concatenate([fs, fa], axis=1), axis=0)
        out.write(f[:, :, [2, 1, 0]])
    out.release()
