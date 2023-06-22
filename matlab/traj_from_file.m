function [pos, rot, vel, waypoint_t, waypoint_pos, waypoint_quat] = traj_from_file( ...
    filename, ...
    scan_t, ...
    scan_window, ...
    do_interp, ...
    interp_fs ...
)

fprintf('Loading %s...\n', filename);
load(filename, 't', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw');

waypoint_t = t;
waypoint_pos = [x, y, z];
waypoint_quat = quaternion(qw, qx, qy, qz);

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
f = waitbar(0, 'Processing trajectory');
for i = 1:M
    waitbar(i/M, f, 'Processing trajectory');
    w_pos = (scan_t(i)-scan_window/2) <= waypoint_t & waypoint_t <= (scan_t(i)+scan_window/2);
    w_vel = (scan_t(i)-scan_window/2) <= waypoint_t(1:end-1) & waypoint_t(1:end-1) <= (scan_t(i)+scan_window/2);
    % Assume pose samples are equally spaced within the scan window
    % h_pos = hann(sum(w_pos));
    % h_vel = hann(sum(w_vel));
    pos(i, :) = mean(waypoint_pos(w_pos, :));
    % pos(i, :) = sum(waypoint_pos(w_pos, :) .* h_pos) ./ sum(h_pos);
    rot(i, :, :) = quat2rotm(meanrot(waypoint_quat(w_pos, :)));
    vel(i, :) = mean(waypoint_vel(w_vel, :));
    % vel(i, :) = sum(waypoint_vel(w_vel, :) .* h_vel) ./ sum(h_vel);
end
delete(f);

end
