"""Create video from output files."""

import os
from tqdm import tqdm
import matplotlib as mpl

from jax import numpy as jnp
import jax
import numpy as np
import cv2
import h5py

from dart import VirtualCameraImage, DartResult


_desc = "Create a video from radar and virtual camera renderings."


def _parse(p):
    p.add_argument(
        "-p", "--path", required=True, help="File to load and render.")
    p.add_argument(
        "-c", "--fourcc", help="Format fourcc code.", default="mp4v")
    p.add_argument("-o", "--out", help="Output file.")
    p.add_argument(
        "-f", "--fps", default=15.625, type=float,
        help="Video framerate; set as 1 / (scan_dt * stride) for 1:1 time.")
    p.add_argument(
        "-s", "--size", type=int, default=512,
        help="Vertical/horizontal size to rescale each plot to.")
    p.add_argument(
        "-b", "--batch", type=int, default=1024, help="Batch size.")
    return p


def _resize(img, size):
    if img.shape[:2] != size:
        return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
    else:
        return img


def _main(args):
    res = DartResult(args.path)

    if args.out is None:
        fname = "{}.video.mp4".format(os.path.basename(args.path))
        args.out = os.path.join(args.path, fname)

    cmap = (jnp.array(mpl.colormaps['viridis'].colors) * 255).astype(jnp.uint8)

    cam = dict(res.open(res.CAMERA))
    rad = res.open(res.RADAR)["rad"]

    mask = h5py.File(os.path.join(res.DATADIR, "trajectory.h5"))['mask']
    gt = np.array(h5py.File(os.path.join(res.DATADIR, "radar.h5"))["rad"][mask])

    valstart = np.min(np.load(os.path.join(args.path, "metadata.npz"))["val"])

    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    out = cv2.VideoWriter(
        args.out, fourcc, args.fps, (args.size * 3, args.size))

    _loadrad = jax.jit(res.colorize_radar)

    frame_idx = 0
    for _ in range(int(np.ceil(rad.shape[0] / args.batch))):
        cam_frames = np.asarray(VirtualCameraImage(
            **{k: v[:args.batch] for k, v in cam.items()}).to_rgb())
        rad_frames = np.asarray(_loadrad(cmap, rad[:args.batch]))
        gt_frames = np.asarray(_loadrad(cmap, gt[:args.batch]))

        cam = {k: v[args.batch:] for k, v in cam.items()}
        rad = rad[args.batch:]
        gt = gt[args.batch:]

        for fc, fr, fg in zip(tqdm(cam_frames), rad_frames, gt_frames):
            fc = _resize(fc, (args.size, args.size))
            fr = _resize(fr, (args.size, args.size))
            fg = _resize(fg, (args.size, args.size))
            f = np.concatenate([fc, fr, fg], axis=1)

            label = "{}:{}".format(
                "t" if frame_idx < valstart else "v", frame_idx)
            cv2.putText(
                f, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2, cv2.LINE_AA)
            frame_idx += 1

            # RGB -> BGR since OpenCV assumes BGR
            out.write(f[:, :, [2, 1, 0]])

    out.release()
