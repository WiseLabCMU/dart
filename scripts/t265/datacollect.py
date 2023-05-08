#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import tables as tb
import pyrealsense2 as rs
import os


TRAJ_BUFSIZE = 200


class Pose(tb.IsDescription):
    t       = tb.Float64Col()
    x       = tb.Float64Col()
    y       = tb.Float64Col()
    z       = tb.Float64Col()
    vx      = tb.Float64Col()
    vy      = tb.Float64Col()
    vz      = tb.Float64Col()
    qw      = tb.Float64Col()
    qx      = tb.Float64Col()
    qy      = tb.Float64Col()
    qz      = tb.Float64Col()


def datacollect():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output', '-o',
        help='Output directory (eg. C:/Users/Administrator/Desktop/dartdata/dataset0',
        default='./'
    )
    args = parser.parse_args()

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.disable_all_streams()
    cfg.enable_stream(rs.stream.pose)
    pipe.start(cfg)
    os.makedirs(args.output)
    outfile = os.path.join(args.output, 't265.h5')
    with tb.open_file(outfile, mode='w', title='Trajectory file') as h5file:
        traj_group = h5file.create_group('/', 'traj', 'Trajectory information')
        pose_table = h5file.create_table(traj_group, 'pose', Pose, 'Pose data')
        pose_count = 0
        start_time = None
        end_time = None
        try:
            while True:
                frames = pipe.poll_for_frames()
                if frames:
                    frame = frames.get_pose_frame()
                    if frame:
                        end_time = frame.timestamp / 1000
                        if not start_time:
                            start_time = end_time
                        pose = frame.get_pose_data()
                        pose_table.row['t'] = frame.timestamp / 1000
                        pose_table.row['x'] = pose.translation.x
                        pose_table.row['y'] = pose.translation.y
                        pose_table.row['z'] = pose.translation.z
                        pose_table.row['vx'] = pose.velocity.x
                        pose_table.row['vy'] = pose.velocity.y
                        pose_table.row['vz'] = pose.velocity.z
                        pose_table.row['qw'] = pose.rotation.w
                        pose_table.row['qx'] = pose.rotation.x
                        pose_table.row['qy'] = pose.rotation.y
                        pose_table.row['qz'] = pose.rotation.z
                        pose_table.row.append()
                        pose_count += 1
                        if pose_count >= TRAJ_BUFSIZE:
                            print(f'Flushing {pose_count} poses.')
                            print(f'Trajectory time: {end_time - start_time}s\n')
                            pose_table.flush()
                            pose_count = 0
        except KeyboardInterrupt:
            pipe.stop()
            print(f'Interrupted.')
            print(f'Flushing {pose_count} poses.')
            print(f'Trajectory time: {end_time - start_time}s\n')
            pose_table.flush()
            pose_count = 0

if __name__ == '__main__':
    datacollect()
