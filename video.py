"""Create video from output files."""

import os
from tqdm import tqdm
import matplotlib as mpl

from scipy.io import loadmat
import numpy as np
import cv2

from dart import VirtualCameraImage
from argparse import ArgumentParser


def _parse():
    p = ArgumentParser()
    p.add_argument("-p", "--path", nargs='+', help="Files to load and render.")
    p.add_argument(
        "-c", "--fourcc", help="Format fourcc code.", default="mp4v")
    p.add_argument("-o", "--out", help="Output file.")
    p.add_argument(
        "-f", "--fps", default=30.0, type=float, help="Video framerate.")
    p.add_argument(
        "-s", "--size", type=int, default=512,
        help="Vertical/horizontal size to rescale each plot to.")
    return p


def _loadrad(path):
    rad = loadmat(path)["rad"]
    p5 = np.nanpercentile(rad, 5)
    p95 = np.nanpercentile(rad, 99.9)
    rad = np.nan_to_num(rad, nan=p5)
    rad = (rad - p5) / (p95 - p5)
    return (mpl.colormaps['viridis'](rad)[:, :, :, :3] * 255).astype(np.uint8)


if __name__ == '__main__':

    args = _parse().parse_args()

    cam_path = [os.path.join(p, "cam_all.mat") for p in args.path]
    rad_path = [os.path.join(p, "pred_all.mat") for p in args.path]
    width = args.size * len(cam_path)

    print("Creating video:")
    print("Top row: {}".format(", ".join(cam_path)))
    print("Bottom row: {}".format(", ".join(cam_path)))

    cam = np.concatenate(
        [VirtualCameraImage.from_file(p).to_rgb() for p in cam_path], axis=2)
    rad = np.concatenate([_loadrad(p) for p in rad_path], axis=2)

    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    out = cv2.VideoWriter(
        args.out, fourcc, args.fps, (width, args.size * 2))
    for fc, fr in tqdm(zip(cam, rad), total=len(cam)):
        fc = cv2.resize(
            fc, (width, args.size), interpolation=cv2.INTER_NEAREST)
        fr = cv2.resize(
            fr, (width, args.size), interpolation=cv2.INTER_NEAREST)
        f = np.concatenate([fc, fr], axis=0)
        # RGB -> BGR since OpenCV assumes BGR
        out.write(cv2.resize(f, (width, args.size * 2))[:, :, [2, 1, 0]])
    out.release()
