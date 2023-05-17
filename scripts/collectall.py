#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import subprocess


RADARCOLLECT = [
    'python',
    'radar/radarcollect.py',
    '-o'
]
T265COLLECT = [
    'python',
    't265/t265collect.py',
    '-o'
]
OPTITRACKCOLLECT = [
    'DartDataCollect/OptitrackCollect/bin/Debug/net6.0/OptitrackCollect.exe',
    '192.168.1.120',
    '192.168.1.77',
    'Multicast'
]


def collect():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output', '-o',
        help='Output directory (eg. C:/Users/Administrator/Desktop/dartdata/dataset0',
        default='./'
    )
    args = parser.parse_args()
    subprocess.Popen(RADARCOLLECT + [args.output])
    subprocess.Popen(T265COLLECT + [args.output])
    subprocess.Popen(OPTITRACKCOLLECT + [args.output])


if __name__ == '__main__':
    collect()
