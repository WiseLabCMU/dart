#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import subprocess
import time
import numpy as np
import os
import socket
import struct
import tables as tb
from datetime import datetime


CMD_DIR = 'C:/ti/mmwave_studio_02_01_01_00/mmWaveStudio/RunTime'
SCRIPT_FILE = 'C:/Users/Administrator/git/dart/scripts/radar/Automation.lua'
PACKET_BUFSIZE = 8192
MAX_PACKET_SIZE = 4096


class Packet(tb.IsDescription):
    t           = tb.Float64Col()
    packet_data = tb.UInt16Col(shape=(728,))
    packet_num  = tb.UInt32Col()
    byte_count  = tb.UInt64Col()


def _parse(p):
    p.add_argument(
        '--static_ip', '-i',
        help='Static IP address (eg 192.168.33.30)',
        default='192.168.33.30')
    p.add_argument(
        '--data_port', '-d', type=int, default=4098,
        help='Port for data stream (eg. 4098)')
    p.add_argument(
        '--timeout', '-t', type=float, default=20
        help='Socket timeout in seconds (eg. 20)')
    p.add_argument(
        '--output', '-o', default=None, help="Output directory. If blank, "
        "creates a folder with the current datetime.")
    return p


def radarcollect(args):
    if args.output is None:
        args.output = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
    os.makedirs(args.output, exist_ok=True)
    outfile = os.path.join(args.output, 'radarpackets.h5')

    cwd = os.getcwd()
    os.chdir(CMD_DIR)
    subprocess.Popen(['mmWaveStudio.exe', '/lua', SCRIPT_FILE])
    os.chdir(cwd)
    print('waiting 56 seconds...')
    time.sleep(56.0)
    print('starting!')

    data_recv = (args.static_ip, args.data_port)

    data_socket = socket.socket(
        socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    data_socket.bind(data_recv)
    data_socket.settimeout(args.timeout)

    with tb.open_file(outfile, mode='w', title='Packet file') as h5file:
        scan_group = h5file.create_group('/', 'scan', 'Scan information')
        packet_table = h5file.create_table(
            scan_group, 'packet', Packet, 'Packet data')
        packet_in_chunk = 0
        num_all_packets = 0
        start_time = time.time()
        try:
            while True:
                packet_num, byte_count, packet_data = _read_data_packet(data_socket)
                curr_time = time.time()
                packet_table.row['t'] = curr_time
                packet_table.row['packet_data'] = packet_data
                packet_table.row['packet_num'] = packet_num
                packet_table.row['byte_count'] = byte_count
                packet_table.row.append()
                packet_in_chunk += 1
                num_all_packets += 1
                if packet_in_chunk >= PACKET_BUFSIZE:
                    print('[t={:.3f}s] Flushing {} packets'.format(
                        curr_time - start_time, packet_in_chunk))
                    packet_table.flush()
                    packet_in_chunk = 0

        except (KeyboardInterrupt, Exception) as e:
            print(e)

        curr_time = time.time()
        print(f'Flushing {packet_in_chunk} packets.')
        print(f'Capture time: {curr_time - start_time}s\n')
        print("Total packets captured ", num_all_packets)
        packet_table.flush()
        packet_in_chunk = 0
        data_socket.close()


def _read_data_packet(data_socket):
    """Helper function to read in a single ADC packet via UDP.

    The format is described in the [DCA1000EVM user guide](
        https://www.ti.com/tool/DCA1000EVM#tech-docs)::

        | packet_num (u4) | byte_count (u6) | data ... |

    The packet_num and byte_count appear to be in little-endian order.

    Returns
    -------
    packet_num: current packet number
    byte_count: byte count of data that has already been read
    data: raw ADC data in current packet
    """
    data = data_socket.recv(MAX_PACKET_SIZE)
    # Little-endian, no padding
    packet_num, byte_count = struct.unpack('<LQ', data[:10] + b'\x00\x00')
    packet_data = np.frombuffer(data[10:], dtype=np.uint16)
    return packet_num, byte_count, packet_data


if __name__ == '__main__':
    args = _parse(argparse.ArgumentParser()).parse_args()
    radarcollect(args)
