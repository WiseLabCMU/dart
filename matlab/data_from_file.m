datadir = 'F:\dartdata';
dataset = 'linear2';

scandir = fullfile(datadir, dataset, 'frames');
trajdir = fullfile(datadir, dataset, 'traj');
outfile = fullfile(datadir, dataset, append(dataset, '.mat'));
jsonfile = fullfile(datadir, dataset, append(dataset, '.json'));
mapfile = fullfile(datadir, dataset, 'map.mat');
dbgfile = fullfile(datadir, dataset, 'dbg.mat');
simfile = fullfile(datadir, dataset, 'simulated.mat');

range_decimation = 8;   % max_range=21m when range_decimation=1
doppler_decimation = 4; % max_velocity=2m/s when doppler_decimation=1
framelen = 256;         % motion during frame should <~2 range bins (.08m)

CHIRPLEN = 512;
% CHIRP_DT = 5e-4;
CHIRP_DT = 1e-3;
% DMAX = 3.7899;
DMAX = 1.8949;
RMAX = 21.5991;
 
bin_doppler = DMAX / framelen;
res_doppler = framelen / doppler_decimation;
min_doppler = -bin_doppler * (res_doppler * 0.5);
max_doppler = bin_doppler * (res_doppler * 0.5 - 1);

bin_range = RMAX / CHIRPLEN;
res_range = CHIRPLEN / range_decimation;
min_range = bin_range * 0.5;
max_range = bin_range * (res_range + 0.5);

scan_window = CHIRP_DT * framelen;

radarjson = struct();
radarjson.theta_lim = deg2rad(90) - 0.001;
radarjson.phi_lim = deg2rad(90) - 0.001;
radarjson.n = 512;
radarjson.k = 256;
radarjson.r = [min_range, max_range, res_range];
radarjson.d = [min_doppler, max_doppler, res_doppler];
jsonstring = jsonencode(radarjson, 'PrettyPrint', true);
writelines(jsonstring, jsonfile);

map = gen_map_linear();
x = map.x;
y = map.y;
z = map.z;
v = map.v;
cx = map.cx;
cy = map.cy;
cz = map.cz;

all_t = [];
all_rad = [];
all_pos = [];
all_rot = [];
all_vel = [];
all_wp_t = [];
all_wp_pos = [];
all_wp_quat = quaternion;
scanfiles = sort(string({dir(fullfile(scandir, '*.mat')).name}));
for i = 1:length(scanfiles)
    scanfile = fullfile(scandir, scanfiles(i));
    [~, filenameNoExt, ~] = fileparts(scanfile);
    trajfile = fullfile(trajdir, append(filenameNoExt, '.txt'));
    
    [scan_t, rad] = scans_from_file( ...
        scanfile, ...
        range_decimation, ...
        doppler_decimation, ...
        framelen ...
        );
    
    [pos, rot, vel, wp_t, wp_pos, wp_quat] = traj_from_file(trajfile, scan_t, scan_window);

    all_t = [all_t; scan_t];
    all_rad = [all_rad; rad];
    all_pos = [all_pos; pos];
    all_rot = [all_rot; rot];
    all_vel = [all_vel; vel];
    all_wp_t = [all_wp_t; wp_t];
    all_wp_pos = [all_wp_pos; wp_pos];
    all_wp_quat = [all_wp_quat; wp_quat];
end
t = all_t;
rad = all_rad;
pos = all_pos;
rot = all_rot;
vel = all_vel;
wp_t = all_wp_t;
wp_pos = all_wp_pos;
wp_quat = all_wp_quat;
clear all_t all_rad all_pos all_rot all_vel all_wp_t all_wp_pos all_wp_quat;
save(outfile, 't', 'rad', 'pos', 'rot', 'vel', '-v7.3');
save(mapfile, 'x', 'y', 'z', 'v', 'cx', 'cy', 'cz', '-v7.3');
save(dbgfile, '-v7.3');
