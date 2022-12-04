map = gen_map();
[traj, timestamp] = gen_traj();
volshow(map.v);

x = map.x;
y = map.y;
z = map.z;
v = map.v;
[position, orientation, velocity, acceleration, angularVelocity] = traj.lookupPose(timestamp);
orientation = permute(quat2rotm(orientation), [3 1 2]);

save('../data/map', 'x', 'y', 'z', 'v');
save('../data/traj', 'timestamp', 'position', 'orientation', 'velocity', 'acceleration', 'angularVelocity');
