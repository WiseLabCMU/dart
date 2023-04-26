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

b = zeros(chirplen, framelen, numframes);
for i = 1:numframes
    startchirp = stride * (i - 1) + 1;
    b(:, :, i) = squeeze(frames(startchirp : startchirp + framelen - 1, 1, 1, :)).';
end
fff = fft2(b);
fff(:, 1, :) = fff(:, 1, :) - median(fff(:, 1, :), 3);

res_doppler = framelen / doppler_decimation;
res_range = chirplen / range_decimation;
fff = circshift(fff, floor(res_doppler / 2), 2); % may have inconsistent phase without fftshift
fff = permute(fff, [3 1 2]);

scans = abs(fff(:, 1:res_range, 1:res_doppler));

end
