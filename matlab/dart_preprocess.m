% ------------------- PARAMS ------------------------------- %

DATADIR = '/media/john/HEADCOUNT/dartdata';
DATASET = 'cichall';
BATCH_SIZE = 50000;

USE_T265 = true;
FORCE_REPROCESS_TRAJ = true;
INTERP_TRAJ = false;
INTERP_TRAJ_FS = 200;

range_decimation = 4;      % max_range=21m when range_decimation=1
doppler_decimation = 1;    % max_velocity=2m/s when doppler_decimation=1
framelen = 256;
stride = 64;

CHIRPLEN = 512;
CHIRP_DT = 1e-3;
DMAX = 1.8949;
RMAX = 21.5991;

GEN_MAP = false;

if USE_T265
    LOCAL_TFORM = [ 0,  1,  0,  0;
                    0,  0, -1,  0;
                   -1,  0,  0,  0;
                    0,  0,  0,  1];
else
    LOCAL_TFORM = [ 1,  0,  0,  0;
                    0,  0,  1,  0;
                    0, -1,  0,  0;
                    0,  0,  0,  1];
end

GLOBAL_TFORM = [ 1,  0,  0,  0;
                 0,  0, -1,  0;
                 0,  1,  0,  0;
                 0,  0,  0,  1];

% ---------------------------------------------------------- %

% radarpacketsfile = fullfile(DATADIR, DATASET, 'radarpackets.h5');
t265file = fullfile(DATADIR, DATASET, 't265.h5');
optitrackfile = fullfile(DATADIR, DATASET, 'optitrack.txt');

scanfile = fullfile(DATADIR, DATASET, 'frames.h5');
trajfile = fullfile(DATADIR, DATASET, 'traj.mat');
outfile = fullfile(DATADIR, DATASET, append(DATASET, '.mat'));
jsonfile = fullfile(DATADIR, DATASET, append(DATASET, '.json'));
mapfile = fullfile(DATADIR, DATASET, 'map.mat');
dbgfile = fullfile(DATADIR, DATASET, 'dbg.mat');

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

if ~exist(trajfile, 'file') || FORCE_REPROCESS_TRAJ
    if USE_T265
        preprocess_t265(t265file, trajfile);
    else
        preprocess_optitrack(optitrackfile, trajfile);
    end
end

nrows = h5info(scanfile).Groups.Datasets.Dataspace.Size;
nbatches = ceil(nrows / BATCH_SIZE);
scan_t = [];
rad = [];
for b = 0 : nbatches - 1
    fprintf('Batch %d/%d\n', b + 1, nbatches);
    [new_scan_t, new_rad] = timed_scans_from_file( ...
        scanfile, ...
        range_decimation, ...
        doppler_decimation, ...
        framelen, ...
        stride, ...
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
    LOCAL_TFORM, ...
    GLOBAL_TFORM, ...
    INTERP_TRAJ, ...
    INTERP_TRAJ_FS ...
);
t = scan_t;

naan = isnan(pos(:,1));
t(naan) = [];
rad(naan, :, :) = [];
pos(naan, :) = [];
rot(naan, :, :) = [];
vel(naan, :) = [];

save(outfile, 't', 'rad', 'pos', 'rot', 'vel', '-v7.3');
save(dbgfile, '-v7.3');
