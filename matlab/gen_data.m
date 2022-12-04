map = gen_map();
[traj, t_sample] = gen_traj();
volshow(map.v);

x = map.x;
y = map.y;
z = map.z;
v = map.v;

save('../data/mapx', 'x');
save('../data/mapy', 'y');
save('../data/mapz', 'z');
save('../data/mapv', 'v');
