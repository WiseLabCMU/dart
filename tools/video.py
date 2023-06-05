"""Create video from output files."""

import os
from tqdm import tqdm
import json
import matplotlib as mpl

from dart.dataset import load_arrays
import numpy as np
import cv2

from dart import VirtualCameraImage


_desc = "Create a video from radar and virtual camera renderings."


def _parse(p):
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
    rad = load_arrays(path)["rad"]
    p5 = np.nanpercentile(rad, 5)
    p95 = np.nanpercentile(rad, 99.9)
    rad = np.nan_to_num(rad, nan=p5)
    rad = (rad - p5) / (p95 - p5)
    return (mpl.colormaps['viridis'](rad)[:, :, :, :3] * 255).astype(np.uint8)


def _get_dataset(path):
    with open(os.path.join(path, "metadata.json")) as f:
        cfg = json.load(f)
    return cfg["dataset"]["path"]


def _resize(img, size):
    return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)


def _main(args):

    if args.out is None:
        args.out = os.path.join(
            args.path[0], "video.mp4")

    cam_path = [os.path.join(p, "cam_all.mat") for p in args.path]
    rad_path = [os.path.join(p, "pred_all.mat") for p in args.path]
    dataset_path = [_get_dataset(p) for p in args.path]
    height = args.size * len(cam_path)

    print("Creating video:")
    print("Left column: {}".format(", ".join(cam_path)))
    print("Middle column: {}".format(", ".join(cam_path)))
    print("Right column: {}".format(", ".join(dataset_path)))

    cam = np.concatenate(
        [VirtualCameraImage.from_file(p).to_rgb() for p in cam_path], axis=1)
    rad = np.concatenate([_loadrad(p) for p in rad_path], axis=1)
    gt = np.concatenate([_loadrad(p) for p in dataset_path], axis=1)

    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    out = cv2.VideoWriter(args.out, fourcc, args.fps, (args.size * 3, height))
    for fc, fr, fg in tqdm(zip(cam, rad, gt), total=len(cam)):
        fc = _resize(fc, (args.size, height))
        fr = _resize(fr, (args.size, height))
        fg = _resize(fg, (args.size, height))
        f = np.concatenate([fc, fr, fg], axis=1)
        # RGB -> BGR since OpenCV assumes BGR
        out.write(cv2.resize(f, (args.size * 3, height))[:, :, [2, 1, 0]])
    out.release()
