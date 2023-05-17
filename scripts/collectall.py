#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import subprocess


def collect():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output', '-o',
        help='Output directory (eg. C:/Users/Administrator/Desktop/dartdata/dataset0',
        default='./'
    )
    args = parser.parse_args()
    subprocess.Popen(['python', 'radar/radarcollect.py', '-o', args.output])
    subprocess.Popen(['python', 't265/t265collect.py', '-o', args.output])


if __name__ == '__main__':
    collect()
