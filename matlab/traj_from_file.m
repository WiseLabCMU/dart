function [pos, rot, vel, waypoints] = traj_from_file(filename, scan_t)

traj = jsondecode(fileread(filename));
N = size(traj, 1);

waypoint_ts = zeros(N, 1);
waypoints = zeros(N, 3);
dirs = zeros(3, 3, N);

f = waitbar(0, 'Loading trajectory');
for p = 1:N
    pose = traj(p).data;
    dt = datetime(pose.timestamp, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS''Z''');
    waypoint_ts(p) = posixtime(dt);
    waypoints(p, 1) = pose.position.x;
    waypoints(p, 2) = -pose.position.z;
    waypoints(p, 3) = pose.position.y;
    mocap_rot = quat2rotm(quaternion(pose.rotation.w, pose.rotation.x, -pose.rotation.z, pose.rotation.y));
    dirs(:, :, p) = axang2rotm([0, 0, 1, deg2rad(-90)]) * mocap_rot;
    f = waitbar(p/N, f, 'Loading trajectory');
end
close(f);

traj = waypointTrajectory(waypoints, waypoint_ts, Orientation=dirs, ReferenceFrame='ENU');

[pos, rot, vel, ~, ~] = traj.lookupPose(scan_t);
rot = permute(quat2rotm(rot), [3 2 1]); % lookupPose provides world->body, so we need to invert

end
