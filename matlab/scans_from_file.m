function [timestamps, scans] = scans_from_file( ...
    filename, ...
    range_decimation, ...
    doppler_decimation, ...
    framelen, ...
    doplot)

if ~exist('doplot', 'var')
    doplot = false;
end

fprintf('Loading %s...\n', filename);
load(filename, 'frames_real', 'frames_imag', 'start_time', 'end_time');
frames = complex(frames_real, frames_imag);

fprintf('Processing %s...\n', filename);
chirplen = size(frames, 4); % before decmiation
numframes = floor(size(frames, 1) / framelen);
numsamples = numframes * framelen;

t_start = posixtime(datetime(start_time, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS''Z'''));
t_start = t_start - 5*60*60;
t_end = posixtime(datetime(end_time, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS''Z'''));
t_end = t_end - 5*60*60;
ts = (t_end - t_start) / numframes;

timestamps = (t_start+ts/2 : ts : t_end-ts/2).';

a = squeeze(frames(1:numsamples, 1, 1, :));
b = reshape(a.', chirplen, framelen, []);

CHIRPLEN = 512;
DMAX = 3.7899;
RMAX = 21.5991;

bin_doppler = DMAX / framelen;
res_doppler = framelen / doppler_decimation;
min_doppler = -bin_doppler * (res_doppler * 0.5);
max_doppler = bin_doppler * (res_doppler * 0.5 - 1);

bin_range = RMAX / CHIRPLEN;
res_range = CHIRPLEN / range_decimation;
min_range = bin_range * 0.5;
max_range = bin_range * (res_range + 0.5);

framelen_dec = framelen / doppler_decimation;
chirplen_dec = chirplen / range_decimation;
u = framelen/2 + (-framelen_dec/2+1 : framelen_dec/2);
v = 1:chirplen_dec;
x = linspace(min_doppler, max_doppler, res_doppler);
y = linspace(min_range, max_range, res_range);
c = permute(fftshift(fft2(b), 2), [3 1 2]);
c = c(:, v, u);

scans = abs(c);

if doplot
    for i = 1:numframes
        imcomplex(x, y, squeeze(abs(c(i, :, :))));
        pause(ts);
    end
end


end
