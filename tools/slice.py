"""Render MRI-style slices, and write each slice to a frame in a video."""

import os
from tqdm import tqdm

from dart import DartResult
import numpy as np
import cv2


def _parse(p):
    p.add_argument("-p", "--path", help="File to load and render.")
    p.add_argument(
        "-c", "--fourcc", help="Format fourcc code.", default="mp4v")
    p.add_argument("-o", "--out", help="Output file.")
    p.add_argument(
        "-f", "--fps", default=10.0, type=float, help="Video framerate.")
    p.add_argument(
        "-r", "--radius", help="Smoothing window radius.", default=0, type=int)
    p.add_argument(
        "--axis", help="Slicing axis (x=0, y=1, z=2).", type=int, default=2)
    return p


def _main(args):
    if args.out is None:
        fname = "{}.slice.{}.mp4".format(
            os.path.basename(args.path), "xyz"[args.axis])
        args.out = os.path.join(args.path, fname)

    res = DartResult(args.path)
    mapfile = res.load(DartResult.MAP)

    sigma = res.colorize_map(
        mapfile["sigma"], conv=args.radius * 2, sigma=True, clip=(1.0, 99.0))
    alpha = res.colorize_map(
        -mapfile["alpha"], conv=args.radius * 2, sigma=False, clip=(1.0, 99.0))

    lower = mapfile["lower"]
    upper = mapfile["upper"]

    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    if args.axis == 2:
        shape = [sigma.shape[1], sigma.shape[0]]
    elif args.axis == 1:
        shape = [sigma.shape[2], sigma.shape[0]]
    else:
        shape = [sigma.shape[2], sigma.shape[1]]

    out = cv2.VideoWriter(
        args.out, fourcc, args.fps, (shape[0] * 2, shape[1]))

    for i in tqdm(range(sigma.shape[args.axis])):
        if args.axis == 2:
            fs = sigma[:, :, i, :]
            fa = alpha[:, :, i, :]
        elif args.axis == 1:
            fs = sigma[:, i, :, :]
            fa = alpha[:, i, :, :]
        else:
            fs = sigma[i, :, :, :]
            fa = alpha[i, :, :, :]

        f = np.concatenate([fs, fa], axis=1)

        z = lower[args.axis] + (upper[args.axis] - lower[args.axis]) * (
            (i + args.radius) / (sigma.shape[args.axis] + args.radius * 2))
        cv2.putText(
            f, "[{:03}] {:.1f}m".format(i + args.radius, z), (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        out.write(f[:, :, [2, 1, 0]])
    out.release()
