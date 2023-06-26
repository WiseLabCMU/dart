"""Prepare ground truth map."""

import os
import math
from tqdm import tqdm

from plyfile import PlyData
import numpy as np

_desc = "Create ground truth map.npz file from point cloud (.ply)."


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset (directory) path.")
    p.add_argument("-b", "--batch", default=1024 * 1024, help="Batch size.")
    p.add_argument(
        "--lower", default=[-4.0, -4.0, -1.0], nargs='+', type=float,
        help="Lower coordinate (x, y, z).")
    p.add_argument(
        "--upper", default=[4.0, 4.0, 1.0], nargs='+', type=float,
        help="Upper coordinate (x, y, z).")
    p.add_argument(
        "--resolution", default=50.0, type=float,
        help="Grid resolution in grid cells per meter.")
    return p


def _main(args):
    data = PlyData.read(os.path.join(args.path, "points.ply"))

    x = data['vertex']['x']
    y = data['vertex']['y']
    z = data['vertex']['z']
    size = [
        math.ceil((u - lw) * args.resolution)
        for lw, u in zip(args.lower, args.upper)]
    grid = np.zeros(size, dtype=bool)

    for _ in tqdm(range(math.ceil(x.shape[0] / args.batch))):
        ix = ((x[:args.batch] - args.lower[0]) * args.resolution).astype(int)
        iy = ((y[:args.batch] - args.lower[1]) * args.resolution).astype(int)
        iz = ((z[:args.batch] - args.lower[2]) * args.resolution).astype(int)

        mask = (
            (ix > 0) & (ix < size[0])
            & (iy > 0) & (iy < size[1])
            & (iz > 0) & (iz < size[2]))

        grid[ix[mask], iy[mask], iz[mask]] = True
        x = x[args.batch:]
        y = y[args.batch:]
        z = z[args.batch:]

    np.savez(
        os.path.join(args.path, "map.npz"), grid=grid,
        lower=np.array(args.lower),
        upper=np.array(args.upper))
