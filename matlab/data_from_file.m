trajfile = 'D:\CupData\pose-4.txt';
scanfile = 'D:\CupData\cup-4.mat';

range_decimation = 16;
doppler_decimation = 8;
framelen = 1024;

[t, rad] = scans_from_file( ...
    scanfile, ...
    range_decimation, ...
    doppler_decimation, ...
    framelen);

[pos, rot, vel, waypoints] = traj_from_file(trajfile, t);

save('../data/frames4', 't', 'rad', 'pos', 'rot', 'vel');
