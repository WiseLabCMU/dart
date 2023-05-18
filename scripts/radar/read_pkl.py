#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import organizer_copy as org
import tables as tb
from scipy.io import savemat
import sys


def parse(args):
    file_name = sys.argv[1]
    file_root = file_name[:-4]

    f = open('./' + file_name,'rb')
    print(f)
    s = pickle.load(f)

    # o = org.Organizer(s, 64, 4, 3, 512)
    o = org.Organizer(s, 1, 4, 3, 512)
    frames = o.organize()

    print(frames.shape)

    savemat(file_root+'.mat',{'frames':frames, 'start_time':s[3], 'end_time':s[4]})

    to_save = {'frames':frames, 'start_time':s[3], 'end_time':s[4], 'num_frames':len(frames)}

    with open('./' + file_root + '_read.pkl', 'wb') as f:
        pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output', '-o',
        help='Output directory (eg. C:/Users/Administrator/Desktop/dartdata/dataset0',
        default='./'
    )
    args = parser.parse_args()
    parse(args)


