#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import time


CMD_DIR = 'C:/ti/mmwave_studio_02_01_01_00/mmWaveStudio/RunTime'
SCRIPT_FILE = 'C:/Users/Administrator/git/dart/collect/Automation.lua'


def _parse(p):
    p.add_argument(
        '--static_ip', '-i',
        help='Static IP address (eg 192.168.33.30)',
        default='192.168.33.30')
    p.add_argument(
        '--data_port', '-d', type=int, default=4098,
        help='Port for data stream (eg. 4098)')
    p.add_argument(
        '--config_port', '-c', type=int, default=4096,
        help='Port for config stream (eg. 4096)')
    p.add_argument(
        '--timeout', '-t', type=float, default=20,
        help='Socket timeout in seconds (eg. 20)')
    p.add_argument(
        '--output', '-o', default=None, help="Output directory. If blank, "
        "creates a folder with the current datetime.")
    return p


if __name__ == '__main__':

    cwd = os.getcwd()
    os.chdir(CMD_DIR)
    subprocess.Popen(['mmWaveStudio.exe', '/lua', SCRIPT_FILE])
    os.chdir(cwd)
    print('waiting 56 seconds...')
    time.sleep(56.0)
    print('starting!')

    subprocess.Popen(["wsl", "python3", "collect.py"] + sys.argv[1:])
