trajfile = 'D:\CupData\pose.txt';
scanfile = 'D:\CupData\cup.mat';

range_decimation = 16;
doppler_decimation = 8;
framelen = 1024;

[t, rad] = scans_from_file( ...
    scanfile, ...
    range_decimation, ...
    doppler_decimation, ...
    framelen);

[pos, rot, vel] = traj_from_file(trajfile, t);

save('../data/frames', 't', 'rad', 'pos', 'rot', 'vel');
