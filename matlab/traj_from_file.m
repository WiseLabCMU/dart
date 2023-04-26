function [pos, rot, vel, waypoint_t, waypoint_pos, waypoint_quat] = traj_from_file( ...
    filename, ...
    scan_t, ...
    scan_window, ...
    do_interp, ...
    interp_fs ...
)

T_OFFSET = 1674069448.98683;

fprintf('Loading %s...\n', filename);
load(filename, 'rt', 't', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw');

waypoint_t = t + T_OFFSET;
waypoint_pos = [x, -z, y];
waypoint_quat = quaternion(qw, qx, -qz, qy);

if do_interp
    interp_t = (waypoint_t(1) : 1 / interp_fs : waypoint_t(end)).';
    waypoint_pos = interp1(waypoint_t, waypoint_pos, interp_t);
    waypoint_t = interp_t;
    n = size(waypoint_pos, 1);
    % Don't support orientation when doing interpolation
    waypoint_quat = quaternion(ones(n, 1), zeros(n, 1), zeros(n, 1), zeros(n, 1));    
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
