function preprocess_t265(infile, outfile)

fprintf('Converting %s...\n', infile);

traj = h5read(infile, '/traj/pose');
rt = traj.t;
t = traj.t;
x = traj.x;
y = traj.y;
z = traj.z;
qx = traj.qx;
qy = traj.qy;
qz = traj.qz;
qw = traj.qw;

save(outfile, 'rt', 't', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', '-v7.3');

end