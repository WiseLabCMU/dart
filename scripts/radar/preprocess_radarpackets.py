#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import organizer_copy as org
import os
import tables as tb
from scipy.io import savemat


BATCH_SIZE = 500000


def preprocess(args):
    infile = os.path.join(args.dir, 'radarpackets.h5')
    outfile = os.path.join(args.dir, 'frames.mat')
    with tb.open_file(infile) as f:
        packet_table = f.root.scan.packet
        print('Loading packets...')
        data = [row['packet_data'] for row in packet_table.iterrows()]
        print('Loading packet_num...')
        num = [row['packet_num'] for row in packet_table.iterrows()]
        print(f'Total packets: {num[0]} to {num[-1]}')
        print('Loading byte_count...')
        count = [row['byte_count'] for row in packet_table.iterrows()]
        print('Loading timestamps...')
        timestamps = [row['t'] for row in packet_table.iterrows()]
        print(f'Total time: {timestamps[-1] - timestamps[0]}s')
        print('Organizing frames...')
        framelist = []
        frametimelist = []
        for b in range(1,2):#range(len(data) // BATCH_SIZE + 1):
            print(f'batch {b+1} / {len(data) // BATCH_SIZE + 1}')
            start = b * BATCH_SIZE
            end = min((b + 1) * BATCH_SIZE, len(data))
            o = org.Organizer((data[start:end], num[start:end], count[start:end]),
                               timestamps[start:end], 1, 4, 3, 512)
            new_frames, new_frametimes = o.organize()
            framelist.append(new_frames)
            frametimelist.append(new_frametimes)
    frames = np.concatenate(framelist, axis=0)
    frametimes = np.concatenate(frametimelist, axis=0)
    print(frames.shape)
    savemat(outfile, {'frames': frames, 'frametimes': frametimes})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dir', '-d',
        help='Working directory (eg. C:/Users/Administrator/Desktop/dartdata/dataset0',
        default='./'
    )
    args = parser.parse_args()
    preprocess(args)


