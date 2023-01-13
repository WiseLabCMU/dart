function [pos, rot, vel] = traj_from_file(filename, scan_t)

%TODO interpolate from traj_t to scan_t
%TODO compute velocity

traj = jsondecode(fileread(filename));
N = size(traj, 1);

traj_t = zeros(N, 1);
pos = zeros(N, 3);
rot = zeros(N, 3, 3);
vel = zeros(N, 3);
f = waitbar(0, 'Loading trajectory');
for p = 1:N
    pose = traj(p).data;
    dt = datetime(pose.timestamp, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS''Z''');
    traj_t(p) = posixtime(dt);
    pos(p, 1) = pose.position.x;
    pos(p, 2) = pose.position.y;
    pos(p, 3) = pose.position.z;
    rot(p, :, :) = quat2rotm(quaternion(pose.rotation.w, pose.rotation.x, pose.rotation.y, pose.rotation.z));
    f = waitbar(p/N, f, 'Loading trajectory');
end
close(f);

end
