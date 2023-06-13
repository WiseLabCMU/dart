"""Create video from output files."""

import os
from tqdm import tqdm
import matplotlib as mpl

from jax import numpy as jnp
import jax
import numpy as np
import cv2

from dart import VirtualCameraImage, DartResult
from dart.jaxcolors import colormap


_desc = "Create a video from radar and virtual camera renderings."


def _parse(p):
    p.add_argument(
        "-p", "--path", required=True, help="File to load and render.")
    p.add_argument(
        "-c", "--fourcc", help="Format fourcc code.", default="mp4v")
    p.add_argument("-o", "--out", help="Output file.")
    p.add_argument(
        "-f", "--fps", default=30.0, type=float, help="Video framerate.")
    p.add_argument(
        "-s", "--size", type=int, default=512,
        help="Vertical/horizontal size to rescale each plot to.")
    return p


def _tile(videos):
    videos_unpack = [videos[:, :, :, i] for i in range(videos.shape[3])]
    left = jnp.concatenate(videos_unpack[:4], axis=1)
    right = jnp.concatenate(videos_unpack[4:], axis=1)
    return jnp.concatenate([left, right], axis=2)


@jax.jit
def _loadrad(cmap, rad):
    p5 = jnp.nanpercentile(rad, 5)
    p95 = jnp.nanpercentile(rad, 99.9)
    rad = jnp.nan_to_num(rad, nan=p5).astype(jnp.float32)
    rad = (rad - p5) / (p95 - p5)
    colors = (colormap(cmap, rad) * 255).astype(jnp.uint8)
    if len(rad.shape) == 4:
        return _tile(colors)
    else:
        return colors


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
    print(args.out)

    cmap = jnp.array(mpl.colormaps['viridis'].colors)

    import time

    start = time.time()
    cam = np.asarray(
        VirtualCameraImage(**res.load(DartResult.CAMERA)).to_rgb())
    print("loaded cam: {:.3f}gb ({:.3f}s)".format(
        cam.size / 1000 / 1000 / 1000, time.time() - start))

    start = time.time()
    rad = np.asarray(
        _loadrad(cmap, res.load(DartResult.RADAR, keys=["rad"])["rad"]))
    print("loaded rad.pred: {:.3f}gb ({:.3f}s)".format(
        rad.size / 1000 / 1000 / 1000, time.time() - start))

    start = time.time()
    gt = np.asarray(_loadrad(cmap, res.data(keys=["rad"])["rad"]))
    print("loaded rad.gt: {:.3f}gb ({:.3f}s)".format(
        gt.size / 1000 / 1000 / 1000, time.time() - start))

    validx = np.zeros(cam.shape[0], dtype=bool)
    valset = np.load(os.path.join(args.path, "metadata.npz"))["validx"]
    validx[valset] = True

    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    out = cv2.VideoWriter(
        args.out, fourcc, args.fps, (args.size * 3, args.size))
    for i, (fc, fr, fg) in enumerate(tqdm(zip(cam, rad, gt), total=len(cam))):
        fc = _resize(fc, (args.size, args.size))
        fr = _resize(fr, (args.size, args.size))
        fg = _resize(fg, (args.size, args.size))
        f = np.concatenate([fc, fr, fg], axis=1)

        cv2.putText(
            f, "{}{}".format("v" if validx[i] else "t", i), (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # RGB -> BGR since OpenCV assumes BGR
        out.write(f[:, :, [2, 1, 0]])
    out.release()
