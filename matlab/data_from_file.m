clear; clc; close all;

trajfile = 'E:\dartdata\CupData\pose.txt';
scanfile = 'E:\dartdata\CupData\cup.mat';

[timestamp, position, orientation, velocity, acceleration, angularVelocity] = traj_from_file(trajfile);
scans = scans_from_file(scanfile);

save('../data/map', 'x', 'y', 'z', 'v');
save('../data/traj', 'timestamp', 'position', 'orientation', 'velocity', 'acceleration', 'angularVelocity');
