function [timestamps, scans] = scans_from_file( ...
    filename, ...
    range_decimation, ...
    doppler_decimation, ...
    framelen ...
    )

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

fff = fft2(b);
fff(:, 1, :) = fff(:, 1, :) - median(fff(:, 1, :), 3);
fff = circshift(fff, framelen_dec / 2, 2); % may have inconsistent phase without fftshift
c = permute(fff, [3 1 2]);
c = c(:, 1:chirplen_dec, 1:framelen_dec);

scans = abs(c);

end
