function [timestamps, scans] = timed_scans_from_file( ...
    filename, ...
    range_decimation, ...
    doppler_decimation, ...
    framelen, ...
    stride, ...
    start_row, ...
    num_rows ...
)

fprintf('Loading %s...\n', filename);
data = h5read(filename, '/radar/frame', start_row, num_rows);
frametimes = data.t;
frames = permute(complex(data.frames_real, data.frames_imag), [4, 3, 2, 1]);
clear data;

fprintf('Processing %s...\n', filename);
chirplen = size(frames, 4); % before decmiation
numchirps_in = size(frames, 1);

numframes = floor((numchirps_in - framelen) / stride) + 1;

timestamps = zeros(numframes, 1);
scans = zeros(chirplen, framelen, 8, numframes);
for i = 1:numframes
    startchirp = stride * (i - 1) + 1;
    chirp_idx = startchirp : startchirp + framelen - 1;
    timestamps(i) = mean(frametimes(chirp_idx));
    scans(:, :, 1, i) = squeeze(frames(chirp_idx, 1, 1, :)).';
    scans(:, :, 2, i) = squeeze(frames(chirp_idx, 1, 2, :)).';
    scans(:, :, 3, i) = squeeze(frames(chirp_idx, 1, 3, :)).';
    scans(:, :, 4, i) = squeeze(frames(chirp_idx, 1, 4, :)).';
    scans(:, :, 5, i) = squeeze(frames(chirp_idx, 3, 1, :)).';
    scans(:, :, 6, i) = squeeze(frames(chirp_idx, 3, 2, :)).';
    scans(:, :, 7, i) = squeeze(frames(chirp_idx, 3, 3, :)).';
    scans(:, :, 8, i) = squeeze(frames(chirp_idx, 3, 4, :)).';
end
scans = fft2(scans); % range-doppler
scans = fft(scans, [], 3); % azimuth
scans(:, 1, :, :) = scans(:, 1, :, :) - median(scans(:, 1, :, :), 4);

res_doppler = framelen / doppler_decimation;
res_range = chirplen / range_decimation;
scans = circshift(scans, floor(res_doppler / 2), 2);
scans = circshift(scans, 4, 3);
scans = permute(scans, [4 1 2 3]);

scans = abs(scans(:, 1:res_range, 1:res_doppler, :));

end
