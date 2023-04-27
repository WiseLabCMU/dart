function [timestamps, scans] = scans_from_file( ...
    filename, ...
    range_decimation, ...
    doppler_decimation, ...
    framelen, ...
    stride ...
)

fprintf('Loading %s...\n', filename);
load(filename, 'frames_real', 'frames_imag', 'start_time', 'end_time');
frames = complex(frames_real, frames_imag);
clear frames_real frames_imag

fprintf('Processing %s...\n', filename);
chirplen = size(frames, 4); % before decmiation
numchirps_in = size(frames, 1);

t_start = posixtime(datetime(start_time, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS''Z'''));
t_start = t_start - 5*60*60;
t_end = posixtime(datetime(end_time, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS''Z'''));
t_end = t_end - 5*60*60;
chirp_dt = (t_end - t_start) / (numchirps_in - 1);

timestamps = (t_start + chirp_dt * (framelen - 1) / 2 : chirp_dt * stride : t_end - chirp_dt * (framelen - 1) / 2).';
numframes = size(timestamps, 1);
assert(numframes == floor((numchirps_in - framelen) / stride) + 1);

scans = zeros(chirplen, framelen, numframes);
for i = 1:numframes
    startchirp = stride * (i - 1) + 1;
    scans(:, :, i) = squeeze(frames(startchirp : startchirp + framelen - 1, 1, 1, :)).';
end
scans = fft2(scans);
scans(:, 1, :) = scans(:, 1, :) - median(scans(:, 1, :), 3);

res_doppler = framelen / doppler_decimation;
res_range = chirplen / range_decimation;
scans = circshift(scans, floor(res_doppler / 2), 2); % may have inconsistent phase without fftshift
scans = permute(scans, [3 1 2]);

scans = abs(scans(:, 1:res_range, 1:res_doppler));

end
