map = gen_map();
[traj, t_sample] = gen_traj();
volshow(map.v);

x = map.x;
y = map.y;
z = map.z;
v = map.v;

save('../data/map', 'x', 'y', 'z', 'v');
