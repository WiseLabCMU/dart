function [pos, rot, vel, waypoint_t, waypoint_pos, waypoint_quat] = traj_from_file( ...
    filename, ...
    scan_t, ...
    scan_window ...
)

fprintf('Loading %s...\n', filename);
load(filename, 'rt', 't', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw');

waypoint_t = t - mean(t) + mean(rt);
waypoint_pos = [x, -z, y];
waypoint_quat = quaternion(qw, qx, -qz, qy);
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

% tt = t(1);
% first_write = true;
% for i = 2:N
%     pose = jsondecode(s(i)).data;
%     dt = datetime(pose.ts_at_receive, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS''Z''');
%     t(i) = posixtime(dt);
%     cur_x = pose.position.x;
%     cur_y = pose.position.y;
%     cur_z = pose.position.z;
%     while tt < t(i)
%         pct = (tt - t(i - 1)) / (t(i) - t(i - 1));
%         trajjson = struct();
%         trajjson.object_id = 'mmwave-radar';
%         trajjson.data = struct();
%         trajjson.data.position = struct();
%         trajjson.data.position.x = cur_x + pct * (cur_x - last_x);
%         trajjson.data.position.y = cur_y + pct * (cur_y - last_y);
%         trajjson.data.position.z = cur_z + pct * (cur_z - last_z);
%         trajjson.data.rotation = struct();
%         trajjson.data.rotation.x = 0.0;
%         trajjson.data.rotation.y = 0.0;
%         trajjson.data.rotation.z = 0.0;
%         trajjson.data.rotation.w = 1.0;
%         trajjson.data.timestamp = tt;
%         dt = datetime(tt, 'ConvertFrom', 'posixtime', 'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS''Z''');
%         trajjson.data.ts_at_receive = dt;
%         jsonstring = jsonencode(trajjson);
%         if first_write
%             writelines(jsonstring, outfile, 'WriteMode', 'overwrite');
%             first_write = false;
%         else
%             writelines(jsonstring, outfile, 'WriteMode', 'append');
%         end
%         tt = tt + 1 / fs;
%     end
%     last_x = cur_x;
%     last_y = cur_y;
%     last_z = cur_z;
% end

end
