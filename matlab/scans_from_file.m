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

framelen_dec = framelen / doppler_decimation;
chirplen_dec = chirplen / range_decimation;
u = framelen/2 + (-framelen_dec/2 : framelen_dec/2-1);
v = 1:chirplen_dec;
x = linspace(-1.89494428863791/doppler_decimation, 1.89494428863791/doppler_decimation, framelen_dec);
y = (v-1+0.5) * 0.0422;
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
