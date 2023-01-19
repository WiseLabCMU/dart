datadir = 'E:\dartdata';
dataset = 'cubes';

scandir = fullfile(datadir, dataset, 'frames');
trajdir = fullfile(datadir, dataset, 'traj');
outfile = fullfile(datadir, dataset, append(dataset, '.mat'));
jsonfile = fullfile(datadir, dataset, append(dataset, '.json'));

range_decimation = 2;   % max_range=21m when range_decimation=1
doppler_decimation = 2; % max_velocity=2m/s when doppler_decimation=1
framelen = 128;         % motion during frame should <~2 range bins (.08m)
                        % each chirp is .0005s

DMAX = 1.89494428863791;
RMAX = 21.5991;
min_doppler = -DMAX / doppler_decimation;
max_doppler = DMAX / doppler_decimation;
res_doppler = framelen / doppler_decimation;
min_range = RMAX / 512 / 2;
max_range = min_range + RMAX / range_decimation;
res_range = 512 / doppler_decimation;

radarjson = struct();
radarjson.theta_lim = deg2rad(15);
radarjson.phi_lim = deg2rad(60);
radarjson.n = 512;
radarjson.k = 256;
radarjson.r = [min_range, max_range, res_range];
radarjson.d = [min_doppler, max_doppler, res_doppler];
jsonstring = jsonencode(radarjson, 'PrettyPrint', true);
writelines(jsonstring, jsonfile);

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

save(outfile, 't', 'rad', 'pos', 'rot', 'vel', '-v7.3');
