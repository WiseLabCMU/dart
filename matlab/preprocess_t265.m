function preprocess_t265(infile, outfile)

fprintf('Converting %s...\n', infile);

LOCAL_TFORM = [ 0,  1,  0,  0;
                0,  0, -1,  0;
               -1,  0,  0,  0;
                0,  0,  0,  1];
GLOBAL_TFORM = [ 1,  0,  0,  0;
                 0,  0, -1,  0;
                 0,  1,  0,  0;
                 0,  0,  0,  1];

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

[x, y, z, qx, qy, qz, qw] = transform_poses(x, y, z, qx, qy, qz, qw, LOCAL_TFORM, GLOBAL_TFORM);

save(outfile, 'rt', 't', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', '-v7.3');

end