"""Data Preprocessing."""

import h5py
import os
import json
from tqdm import tqdm
import numpy as np

from scipy.ndimage import gaussian_filter1d

from dart.preprocess import AWR1843Boost, AWR1843BoostDataset, Trajectory


_desc = """Preprocess radar and pose data.

Takes a `radarpackets.h5` file containing raw radar packets from
`radarcollect.py`, and a `trajectory.csv` file containing trajectory points
output by cartographer, and creates a `data.h5` file with processed
range-doppler-azimuth images and poses.
"""


def _process_batch(radar: AWR1843Boost, file_batch, traj: Trajectory):
    fields = ["byte_count", "packet_data", "packet_num", "t"]
    packets = {k: file_batch[k] for k in fields}
    dataset = AWR1843BoostDataset.from_packets(packets, radar.frame_size * 2)

    chirps, t_chirp = dataset.as_frames(radar, packets)
    range_doppler, t_image = radar.process_data(chirps, t_chirp)

    t_valid = traj.valid_mask(t_image)
    pose = traj.interpolate(t_image[t_valid])
    pose["t"] = t_image[t_valid]
    return range_doppler[t_valid], pose


def _process(
        path: str, radar: AWR1843Boost, batch_size: int = 1000000,
        overwrite: bool = False, sigma: float = 10.0):

    traj = Trajectory.from_csv(path)
    packetfile = h5py.File(os.path.join(path, "radarpackets.h5"), 'r')
    packet_dataset = packetfile["scan"]["packet"]
    mode = 'w' if overwrite else 'w-'
    outfile = h5py.File(os.path.join(path, "data.h5"), mode)

    with open(os.path.join(path, "sensor.json"), 'w') as f:
        json.dump(radar.to_instrinsics(), f, indent=4)

    rs = radar.image_shape
    range_doppler_azimuth = outfile.create_dataset(
        "rad", (1, *rs), dtype='f2', chunks=(1, *rs), maxshape=(None, *rs))

    total_size = 0
    poses = []
    for _ in tqdm(range(int(np.ceil(packet_dataset.shape[0] / batch_size)))):
        rda, pose = _process_batch(radar, packet_dataset[:batch_size], traj)
        poses.append(pose)

        total_size += rda.shape[0]
        range_doppler_azimuth.resize((total_size, *radar.image_shape))
        range_doppler_azimuth[-rda.shape[0]:] = rda

        packet_dataset = packet_dataset[batch_size:]

    for k in poses[0]:
        val = np.concatenate([p[k] for p in poses])
        if k == 'vel':
            val = gaussian_filter1d(val, sigma=sigma, axis=0)
        outfile.create_dataset(k, data=val)

    packetfile.close()
    outfile.close()


def _parse(p):
    p.add_argument(
        "-p", "--path", help="Dataset path.")
    p.add_argument(
        "-v", "--overwrite", help="Overwrite existing data file.",
        default=False, action='store_true')
    p.add_argument(
        "-b", "--batch", help="Packet processing batch size.",
        type=int, default=1000000)
    p.add_argument(
        "-s", "--smooth", type=float, default=10.0,
        help="Velocity smoothing sigma (for a gaussian filter).")
    return p


def _main(args):

    _process(
        args.path, AWR1843Boost(), batch_size=args.batch,
        overwrite=args.overwrite, sigma=args.smooth)
