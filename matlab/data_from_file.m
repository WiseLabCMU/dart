datadir = 'D:\dartdata';
dataset = 'cup';

scandir = fullfile(datadir, dataset, 'frames');
trajdir = fullfile(datadir, dataset, 'traj');
outfile = fullfile(datadir, dataset, 'data.mat');

all_t = [];
all_rad = [];
all_pos = [];
all_rot = [];
all_vel = [];
scanfiles = sort(string({dir(fullfile(scandir, '*.mat')).name}));
for i = 1:length(scanfiles)
    scanfile = fullfile(scandir, scanfiles(i));
    [~, filenameNoExt, ~] = fileparts(scanfile);
    trajfile = fullfile(trajdir, append(filenameNoExt, '.txt'));
    
    range_decimation = 16;
    doppler_decimation = 8;
    framelen = 1024;
    
    [t, rad] = scans_from_file( ...
        scanfile, ...
        range_decimation, ...
        doppler_decimation, ...
        framelen);
    
    [pos, rot, vel, ~] = traj_from_file(trajfile, t);

    all_t = [all_t; t];
    all_rad = [all_rad; rad];
    all_pos = [all_pos; pos];
    all_rot = [all_rot; rot];
    all_vel = [all_vel; vel];
end
t = all_t;
rad = all_rad;
pos = all_pos;
rot = all_rot;
vel = all_vel;

save(outfile, 't', 'rad', 'pos', 'rot', 'vel');
