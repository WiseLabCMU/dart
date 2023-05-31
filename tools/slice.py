"""Create MRI-style video from map."""

import os
from tqdm import tqdm
import json
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
        "-f", "--fps", default=30.0, type=float, help="Video framerate.")
    p.add_argument(
        "-s", "--size", type=int, default=512,
        help="Vertical/horizontal size to rescale each plot to.")
    return p


def _colorize(x):
    return (mpl.colormaps['viridis'](x)[:, :, :, :3] * 255).astype(np.uint8)


def _loadmap(path):
    data = load_arrays(path)
    return _colorize(data["sigma"]), _colorize(data["alpha"])


def _resize(img, size):
    return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)


def _main(args):
    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    out = cv2.VideoWriter(
        args.out, fourcc, args.fps, (args.size * 2, args.size))

    sigma, alpha = _loadmap(os.path.join(args.path, "map.mat"))
    for i in tqdm(range(sigma.shape[2])):
        fs = _resize(sigma[:, :, i, :], (args.size, args.size))
        fa = _resize(alpha[:, :, i, :], (args.size, args.size))
        f = np.concatenate([fs, fa], axis=1)
        out.write(f[:, :, [2, 1, 0]])
    out.release()
