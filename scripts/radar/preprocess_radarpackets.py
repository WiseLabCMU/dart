#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import organizer_copy as org
import os
import tables as tb


BATCH_SIZE = 500000


class Frame(tb.IsDescription):
    t           = tb.Float64Col()
    frames_real = tb.Int16Col(shape=(3, 4, 512))
    frames_imag = tb.Int16Col(shape=(3, 4, 512))


def preprocess(args):
    infile = os.path.join(args.dir, 'radarpackets.h5')
    outfile = os.path.join(args.dir, 'frames.mat')
    with tb.open_file(infile) as f_in, tb.open_file(outfile, mode='w', title='Radar file') as f_out:
        packet_table = f_in.root.scan.packet
        radar_group = f_out.create_group('/', 'Radar', 'Radar information')
        frame_table = f_out.create_table(radar_group, 'frame', Frame, 'Frame data')
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
        for b in range(len(data) // BATCH_SIZE + 1):
            print(f'batch {b+1} / {len(data) // BATCH_SIZE + 1}')
            start = b * BATCH_SIZE
            end = min((b + 1) * BATCH_SIZE, len(data))
            o = org.Organizer((data[start:end], num[start:end], count[start:end]),
                            timestamps[start:end], 1, 4, 3, 512)
            frames_real, frames_imag, frametimes = o.organize()
            for i in range(frames_real.shape[0]):
                frame_table.row['t'] = frametimes[i]
                frame_table.row['frames_real'] = frames_real[i]
                frame_table.row['frames_imag'] = frames_imag[i]
                frame_table.row.append()
            frame_table.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dir', '-d',
        help='Working directory (eg. C:/Users/Administrator/Desktop/dartdata/dataset0',
        default='./'
    )
    args = parser.parse_args()
    preprocess(args)


