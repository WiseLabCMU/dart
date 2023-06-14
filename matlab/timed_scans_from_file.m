function [timestamps, scans] = timed_scans_from_file( ...
    filename, ...
    range_decimation, ...
    doppler_decimation, ...
    framelen, ...
    stride, ...
    process_azimuth, ...
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

w_d = hann(framelen).';
w_r = hann(chirplen);
w = w_r * w_d;

timestamps = zeros(numframes, 1);
if process_azimuth
    scans = zeros(chirplen, framelen, 8, numframes);
else
    scans = zeros(chirplen, framelen, numframes);
end
for i = 1:numframes
    startchirp = stride * (i - 1) + 1;
    chirp_idx = startchirp : startchirp + framelen - 1;
    timestamps(i) = mean(frametimes(chirp_idx));
    if process_azimuth
        scans(:, :, 1, i) = double(squeeze(frames(chirp_idx, 1, 1, :))).' .* w;
        scans(:, :, 2, i) = double(squeeze(frames(chirp_idx, 1, 2, :))).' .* w;
        scans(:, :, 3, i) = double(squeeze(frames(chirp_idx, 1, 3, :))).' .* w;
        scans(:, :, 4, i) = double(squeeze(frames(chirp_idx, 1, 4, :))).' .* w;
        scans(:, :, 5, i) = double(squeeze(frames(chirp_idx, 3, 1, :))).' .* w;
        scans(:, :, 6, i) = double(squeeze(frames(chirp_idx, 3, 2, :))).' .* w;
        scans(:, :, 7, i) = double(squeeze(frames(chirp_idx, 3, 3, :))).' .* w;
        scans(:, :, 8, i) = double(squeeze(frames(chirp_idx, 3, 4, :))).' .* w;
    else
        scans(:, :, i) = double(squeeze(frames(chirp_idx, 1, 1, :))).' .* w;
    end
end

scans = fft2(scans); % range-doppler
if process_azimuth
    scans = fft(scans, [], 3); % azimuth
    scans(:, [1:2,end], :, :) = scans(:, [1:2,end], :, :) - median(scans(:, [1:2,end], :, :), 4);
else
    scans(:, [1:2,end], :) = scans(:, [1:2,end], :) - median(scans(:, [1:2,end], :), 3);
end

res_doppler = framelen / doppler_decimation;
res_range = chirplen / range_decimation;
scans = circshift(scans, floor(res_doppler / 2), 2);
if process_azimuth
    scans = circshift(scans, 4, 3);
    scans = permute(scans, [4 1 2 3]);
    scans = abs(scans(:, 1:res_range, 1:res_doppler, :));
else
    scans = permute(scans, [3 1 2]);
    scans = abs(scans(:, 1:res_range, 1:res_doppler));
end

end
