"""Create filtered train/val dataset (`JAX_PLATFORM_NAME=cpu` suggested)."""

import os
import h5py
import json

import jax
from jax import numpy as jnp
import numpy as np

from dart import pose, VirtualRadar


def _parse(p):
    p.add_argument("-p", "--path", help="Path to dataset.")
    p.add_argument(
        "--val", default=0.2, type=float,
        help="Validation data holdout proportion.")
    p.add_argument(
        "--norm", default=1.0, type=float, help="Normalization value.")
    p.add_argument(
        "-v", "--overwrite", help="Overwrite existing data file.",
        default=False, action='store_true')
    return p


def _load_data(slam, radar, sensor, norm: float = 1.0):
    """Load poses, images, and doppler data.
    
    Returned data is in (frame, doppler) order, with `images` having additional
    (range, azimuth) axes appended.
    """
    def _load(k):
        return np.array(slam[k])[slam['valid']]  # type: ignore

    # Pre-compute pose information
    v = _load('vel')
    poses = jax.vmap(pose.make_pose)(
        v=v, x=_load('pos'), A=_load('rot'), i=jnp.arange(v.shape[0]))

    # Radar images
    rad = np.array(radar['rad'])[slam['mask']][slam['valid']]  # type: ignore
    images = (np.maximum(rad, 0.0) / norm).astype(np.float16)

    # Reshape to <frame x doppler>, range, azimuth
    ni, nr, nd, na = images.shape
    images_col = np.swapaxes(images, 1, 2).reshape(-1, nr, na)
    poses_col = jax.tree_util.tree_map(
        lambda arr: np.repeat(arr, nd, axis=0), poses)
    doppler = np.tile(sensor.d, ni)
    i_doppler = np.tile(np.arange(len(sensor.d), dtype=np.uint16), ni)
    i_frame = np.repeat(np.arange(ni, dtype=np.uint16), nd)

    return poses_col, {
        "doppler": doppler, "rad": images_col,
        "doppler_idx": i_doppler, "frame_idx": i_frame}


def _main(args):

    if args.overwrite:
        try:
            os.remove(os.path.join(args.path, 'data.h5'))
        except OSError:
            pass
    outfile = h5py.File(os.path.join(args.path, 'data.h5'), 'w')

    radar = h5py.File(os.path.join(args.path, "radar.h5"))
    slam = h5py.File(os.path.join(args.path, "trajectory.h5"))
    with open(os.path.join(args.path, "sensor.json")) as f:
        sensor = VirtualRadar.from_config(**json.load(f))

    poses, data = _load_data(slam, radar, sensor, norm=args.norm)

    # Filter by invalid columns
    psi_min = jax.vmap(sensor.get_psi_min)(d=data["doppler"], pose=poses)
    weight = psi_min / jnp.pi / poses.s
    mask = weight > 0
    print("Valid columns: {}/{}".format(np.sum(mask), mask.shape[0]))
    poses, data = jax.tree_util.tree_map(lambda arr: arr[mask], (poses, data))

    for k, v in data.items():
        outfile.create_dataset(k, data=v)
    outfile.create_dataset("weight", data=weight[mask])
    poses.to_h5file(outfile)
