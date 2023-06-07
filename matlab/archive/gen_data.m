clear; clc; close all;

map = gen_map();
[traj, timestamp] = gen_traj();

x = map.x;
y = map.y;
z = map.z;
v = map.v;
[position, orientation, velocity, acceleration, angularVelocity] = traj.lookupPose(timestamp);
orientation = permute(quat2rotm(orientation), [3 2 1]); % lookupPose provides world->body, so we need to invert

save('../data/map', 'x', 'y', 'z', 'v');
save('../data/traj', 'timestamp', 'position', 'orientation', 'velocity', 'acceleration', 'angularVelocity');

volshow(map.v);
scatter3(position(:,1), position(:,2), position(:,3),'.');
axis equal;
axis vis3d;
