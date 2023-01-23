function [pos, rot, vel, waypoint_t, waypoint_pos, waypoint_quat] = traj_from_file(filename, scan_t, scan_window)

s = readlines(filename);
N = size(s, 1) - 1; % Skip last line to handle EOF
waypoint_t = zeros(N, 1);
waypoint_pos = zeros(N, 3);
waypoint_quat = quaternion(ones(N, 1), zeros(N, 1), zeros(N, 1), zeros(N, 1));
fprintf('Loading %s...\n', filename);
for i = 1:N
    pose = jsondecode(s(i)).data;
    dt = datetime(pose.ts_at_receive, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS''Z''');
    waypoint_t(i) = posixtime(dt);
    waypoint_pos(i, 1) = pose.position.x;
    waypoint_pos(i, 2) = -pose.position.z;
    waypoint_pos(i, 3) = pose.position.y;
    waypoint_quat(i) = quaternion(pose.rotation.w, pose.rotation.x, -pose.rotation.z, pose.rotation.y);
end
waypoint_vel = diff(waypoint_pos) ./ diff(waypoint_t);

M = size(scan_t, 1);
pos = zeros(M, 3);
rot = zeros(M, 3, 3);
vel = zeros(M, 3);
fprintf('Processing %s...\n', filename);
for i = 1:M
    w_pos = (scan_t(i)-scan_window/2) <= waypoint_t & waypoint_t <= (scan_t(i)+scan_window/2);
    w_vel = (scan_t(i)-scan_window) <= waypoint_t & waypoint_t <= (scan_t(i)+scan_window);
    pos(i, :) = mean(waypoint_pos(w_pos, :));
    rot(i, :, :) = quat2rotm(meanrot(waypoint_quat(w_pos, :)));
    vel(i, :) = mean(waypoint_vel(w_vel(1:end-1), :));
end

end
