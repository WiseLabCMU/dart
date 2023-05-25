function [timestamps, scans] = timed_scans_from_file( ...
    filename, ...
    range_decimation, ...
    doppler_decimation, ...
    framelen, ...
    stride ...
)

fprintf('Loading %s...\n', filename);
data = h5read(filename, '/radar/frame');
frametimes = data.t;
frames = permute(complex(data.frames_real, data.frames_imag), [4, 3, 2, 1]);
clear data;

fprintf('Processing %s...\n', filename);
chirplen = size(frames, 4); % before decmiation
numchirps_in = size(frames, 1);

numframes = floor((numchirps_in - framelen) / stride) + 1;

timestamps = zeros(numframes, 1);
scans = zeros(chirplen, framelen, numframes);
for i = 1:numframes
    startchirp = stride * (i - 1) + 1;
    chirp_idx = startchirp : startchirp + framelen - 1;
    timestamps(i) = mean(frametimes(chirp_idx));
    scans(:, :, i) = squeeze(frames(chirp_idx, 1, 1, :)).';
end
scans = fft2(scans);
scans(:, 1, :) = scans(:, 1, :) - median(scans(:, 1, :), 3);

res_doppler = framelen / doppler_decimation;
res_range = chirplen / range_decimation;
scans = circshift(scans, floor(res_doppler / 2), 2); % may have inconsistent phase without fftshift
scans = permute(scans, [3 1 2]);

scans = abs(scans(:, 1:res_range, 1:res_doppler));

end
