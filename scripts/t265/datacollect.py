#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import tables as tb
import pyrealsense2 as rs
import os
# import numpy as np


POSE_BUFSIZE = 200
# CAM_BUFSIZE = 30


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


# class CameraFrame(tb.IsDescription):
#     t       = tb.Float64Col()
#     im_l    = tb.UInt8Col(shape=(800, 848))
#     im_r    = tb.UInt8Col(shape=(800, 848))


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
    # cfg.enable_stream(rs.stream.fisheye, 1)
    # cfg.enable_stream(rs.stream.fisheye, 2)
    pipe.start(cfg)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    outfile = os.path.join(args.output, 't265.h5')
    with tb.open_file(outfile, mode='w', title='Trajectory file') as h5file:
        traj_group = h5file.create_group('/', 'traj', 'Trajectory information')
        pose_table = h5file.create_table(traj_group, 'pose', Pose, 'Pose data')
        # cam_table = h5file.create_table(traj_group, 'camframe', CameraFrame, 'Camera frames')
        pose_count = 0
        # cam_count = 0
        pose_start_time = None
        pose_end_time = None
        # cam_start_time = None
        # cam_end_time = None
        try:
            while True:
                frames = pipe.poll_for_frames()
                if frames:
                    pose_frame = frames.get_pose_frame()
                    if pose_frame:
                        pose_end_time = pose_frame.timestamp / 1000
                        # print(f'pose frame {pose_end_time}')
                        if not pose_start_time:
                            pose_start_time = pose_end_time
                        pose = pose_frame.get_pose_data()
                        pose_table.row['t'] = pose_end_time
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
                        if pose_count >= POSE_BUFSIZE:
                            print(f'Flushing {pose_count} poses.')
                            print(f'Trajectory time: {pose_end_time - pose_start_time}s\n')
                            pose_table.flush()
                            pose_count = 0
                    # cam_frame_l = frames.get_fisheye_frame(1)
                    # cam_frame_r = frames.get_fisheye_frame(2)
                    # if cam_frame_l and cam_frame_r:
                    #     cam_end_time = cam_frame_l.timestamp / 1000
                    #     print(f'cam frame {cam_end_time}')
                    #     if not cam_start_time:
                    #         cam_start_time = cam_end_time
                    #     cam_table.row['t'] = cam_end_time
                    #     cam_table.row['im_l'] = np.asanyarray(cam_frame_l.data)
                    #     cam_table.row['im_r'] = np.asanyarray(cam_frame_r.data)
                    #     cam_table.row.append()
                    #     cam_count += 1
                    #     if cam_count >= CAM_BUFSIZE:
                    #         print(f'Flushing {cam_count} cam frames.')
                    #         print(f'Cam time: {cam_end_time - cam_start_time}s\n')
                    #         cam_table.flush()
                    #         cam_count = 0
                        
        except KeyboardInterrupt:
            pipe.stop()
            print(f'Interrupted.')
            print(f'Flushing {pose_count} poses.')
            print(f'Trajectory time: {pose_end_time - pose_start_time}s\n')
            pose_table.flush()
            pose_count = 0
            # print(f'Flushing {cam_count} cam frames.')
            # print(f'Cam time: {cam_end_time - cam_start_time}s\n')
            # cam_table.flush()
            # cam_count = 0

if __name__ == '__main__':
    datacollect()
