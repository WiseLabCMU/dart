#!/usr/bin/python
# -*- coding: utf-8 -*-

import tables as tb
import pyrealsense2 as rs

TRAJFILE = '../data/traj.h5'


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


if __name__ == '__main__':
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.disable_all_streams()
    cfg.enable_stream(rs.stream.pose)
    pipe.start(cfg)
    with tb.open_file(TRAJFILE, mode='w', title='Trajectory file') as h5file:
        traj_group = h5file.create_group('/', 'traj', 'Trajectory information')
        pose_table = h5file.create_table(traj_group, 'pose', Pose, 'Pose data')
        try:
            while True:
                frames = pipe.poll_for_frames()
                if frames:
                    frame = frames.get_pose_frame()
                    if frame:
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
                        print("Frame #{}".format(frame.frame_number))
        finally:
            pipe.stop()
            pose_table.flush()
