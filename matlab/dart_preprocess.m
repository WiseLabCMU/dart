% ------------------- PARAMS ------------------------------- %

DATADIR = '/media/john/Extreme Pro/dart/';
DATASET = 'cichall-4';
BATCH_SIZE = 50000;

% TRAJ_METHOD = 'optitrack';
% TRAJ_METHOD = 't265';
TRAJ_METHOD = 'cartographer';
INTERP_TRAJ = false;
INTERP_TRAJ_FS = 200;

RANGE_DECIMATION = 4;      % max_range=21m when range_decimation=1
DOPPLER_DECIMATION = 1;    % max_velocity=2m/s when doppler_decimation=1
FRAMELEN = 512;
STRIDE = 128;
PROCESS_AZIMUTH = true;

CHIRPLEN = 512;
CHIRP_DT = 5e-4;
DMAX = 3.7898;
RMAX = 21.5991;

GEN_MAP = false;

if PROCESS_AZIMUTH
    GAIN = 'awr1843boost_az8';
else
    GAIN = 'awr1843boost';
end

% ---------------------------------------------------------- %

t265file = fullfile(DATADIR, DATASET, 't265.h5');
optitrackfile = fullfile(DATADIR, DATASET, 'optitrack.txt');
cartographerfile = fullfile(DATADIR, DATASET, DATASET + ".csv");

scanfile = fullfile(DATADIR, DATASET, 'frames.h5');
trajfile = fullfile(DATADIR, DATASET, 'traj.mat');
outfile = fullfile(DATADIR, DATASET, 'data.mat');
jsonfile = fullfile(DATADIR, DATASET, 'sensor.json');
mapfile = fullfile(DATADIR, DATASET, 'map.mat');
dbgfile = fullfile(DATADIR, DATASET, 'dbg.mat');

bin_doppler = DMAX / FRAMELEN;
res_doppler = FRAMELEN / DOPPLER_DECIMATION;
min_doppler = -bin_doppler * (res_doppler * 0.5);
max_doppler = bin_doppler * (res_doppler * 0.5 - 1);

bin_range = RMAX / CHIRPLEN;
res_range = CHIRPLEN / RANGE_DECIMATION;
% min_range = bin_range * 0.5;
% max_range = bin_range * (res_range + 0.5);
min_range = 0;
max_range = bin_range * res_range;

scan_window = CHIRP_DT * FRAMELEN;

radarjson = struct();
radarjson.theta_lim = deg2rad(90) - 0.001;
radarjson.phi_lim = deg2rad(90) - 0.001;
radarjson.n = 512;
radarjson.k = 256;
radarjson.r = [min_range, max_range, res_range];
radarjson.d = [min_doppler, max_doppler, res_doppler];
radarjson.gain = GAIN;
jsonstring = jsonencode(radarjson, 'PrettyPrint', true);
writelines(jsonstring, jsonfile);

if GEN_MAP
    map = gen_map();
    x = map.x;
    y = map.y;
    z = map.z;
    v = map.v;
    cx = map.cx;
    cy = map.cy;
    cz = map.cz;
    save(mapfile, 'x', 'y', 'z', 'v', 'cx', 'cy', 'cz', '-v7.3');
end

if strcmpi(TRAJ_METHOD, 't265')
    preprocess_t265(t265file, trajfile);
elseif strcmpi(TRAJ_METHOD, 'optitrack')
    preprocess_optitrack(optitrackfile, trajfile);
elseif strcmpi(TRAJ_METHOD, 'cartographer')
    preprocess_cartographer(cartographerfile, trajfile);
else
    error('Unrecognized TRAJ_METHOD: %s', TRAJ_METHOD);
end

nrows = h5info(scanfile).Groups.Datasets.Dataspace.Size;
nbatches = ceil(nrows / BATCH_SIZE);
scan_t = [];
rad = [];
for b = 0 : nbatches - 1
    fprintf('Batch %d/%d\n', b + 1, nbatches);
    [new_scan_t, new_rad] = timed_scans_from_file( ...
        scanfile, ...
        RANGE_DECIMATION, ...
        DOPPLER_DECIMATION, ...
        FRAMELEN, ...
        STRIDE, ...
        PROCESS_AZIMUTH, ...
        b * BATCH_SIZE + 1, ...
        min(BATCH_SIZE, nrows - b * BATCH_SIZE) ...
    );
    scan_t = cat(1, scan_t, new_scan_t);
    rad = cat(1, rad, new_rad);
end

[pos, rot, vel, wp_t, wp_pos, wp_quat] = traj_from_file( ...
    trajfile, ...
    scan_t, ...
    scan_window, ...
    INTERP_TRAJ, ...
    INTERP_TRAJ_FS ...
);
t = scan_t;

naan = isnan(pos(:,1));
t(naan) = [];
if PROCESS_AZIMUTH
    rad(naan, :, :, :) = [];
else
    rad(naan, :, :) = [];
end
pos(naan, :) = [];
rot(naan, :, :) = [];
vel(naan, :) = [];

fprintf('Saving output...\n');
save(outfile, 't', 'rad', 'pos', 'rot', 'vel', '-v7.3');
save(dbgfile, '-v7.3');
