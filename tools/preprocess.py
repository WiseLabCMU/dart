"""Data Preprocessing."""

import h5py
import os
import json
from tqdm import tqdm
import numpy as np

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
    dataset = AWR1843BoostDataset.from_packets(packets, radar.chirp_size)
    return dataset.process_data(radar, traj, packets)


def _process(
        args, path: str, radar: AWR1843Boost, batch_size: int = 1000000,
        overwrite: bool = False, smoothing: float = 2.0):

    traj = Trajectory.from_csv(path)
    packetfile = h5py.File(os.path.join(path, "radarpackets.h5"), 'r')
    packet_dataset = packetfile["scan"]["packet"]
    if overwrite:
        try:
            os.remove(os.path.join(path, 'data.h5'))
        except OSError:
            pass
    outfile = h5py.File(os.path.join(path, "data.h5"), 'w')

    with open(os.path.join(path, "sensor.json"), 'w') as f:
        json.dump(radar.to_instrinsics(), f, indent=4)

    rs = radar.image_shape
    range_doppler_azimuth = outfile.create_dataset(
        "rad", (1, *rs), dtype='f2', chunks=(1, *rs), maxshape=(None, *rs))

    total_size = 0
    _poses = []
    for _ in tqdm(range(int(np.ceil(packet_dataset.shape[0] / batch_size)))):
        try:
            rda, pose = _process_batch(
                radar, packet_dataset[:batch_size], traj)
            _poses.append(pose)

            total_size += rda.shape[0]
            range_doppler_azimuth.resize((total_size, *radar.image_shape))
            range_doppler_azimuth[-rda.shape[0]:] = rda

        except Exception as e:
            print(e)

        packet_dataset = packet_dataset[batch_size:]

    poses = {}
    for k in _poses[0]:
        poses[k] = np.concatenate([p[k] for p in _poses])

    poses['vel_raw'] = poses['vel']
    poses['vel'] = traj.postprocess(
        poses['vel'], poses['speed'], smoothing=smoothing, adjust=args.adjust)

    for k, v in poses.items():
        outfile.create_dataset(k, data=v)
    packetfile.close()
    outfile.close()


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument(
        "-v", "--overwrite", help="Overwrite existing data file.",
        default=False, action='store_true')
    p.add_argument(
        "-b", "--batch", help="Packet processing batch size.",
        type=int, default=1000000)
    p.add_argument(
        "-s", "--smooth", type=float, default=-1.0,
        help="Velocity smoothing sigma (for a gaussian filter).")
    p.add_argument(
        "-a", "--adjust", default=False, action='store_true',
        help="Adjust velocity based on doppler speed estimate.")
    return p


def _main(args):

    _process(
        args, args.path, AWR1843Boost(), batch_size=args.batch,
        overwrite=args.overwrite, smoothing=args.smooth)
