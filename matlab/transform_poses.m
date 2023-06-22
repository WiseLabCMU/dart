function [x, y, z, qx, qy, qz, qw] = transform_poses(x, y, z, qx, qy, qz, qw, local_tform, global_tform)

for i = 1:length(x)
    traj_pose = eye(4);
    traj_pose(1:3, 1:3) = quat2rotm(quaternion(qw(i), qx(i), qy(i), qz(i)));
    traj_pose(1:3, 4) = [x(i); y(i); z(i)];
    dart_pose = global_tform * traj_pose * local_tform;
    x(i) = dart_pose(1, 4);
    y(i) = dart_pose(2, 4);
    z(i) = dart_pose(3, 4);
    quat = rotm2quat(dart_pose(1:3, 1:3));
    qw(i) = quat(1);
    qx(i) = quat(2);
    qy(i) = quat(3);
    qz(i) = quat(4);
end

end

