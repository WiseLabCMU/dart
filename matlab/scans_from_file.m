function [timestamps, scans] = scans_from_file( ...
    filename, ...
    range_decimation, ...
    doppler_decimation, ...
    framelen, ...
    doplot)

if ~exist('doplot', 'var')
    doplot = false;
end

load(filename, 'frames', 'start_time', 'end_time');

dt_start = datetime(start_time, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS''Z''');
dt_end = datetime(end_time, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS''Z''');
duration = posixtime(dt_end) - posixtime(dt_start);

chirplen = size(frames, 4); % before decmiation
numframes = floor(size(frames, 1) / framelen);
numsamples = numframes * framelen;

a = squeeze(frames(1:numsamples, 1, 1, :));
b = reshape(a.', chirplen, framelen, []);

framelen_dec = framelen / doppler_decimation;
chirplen_dec = chirplen / range_decimation;
u = framelen/2 + (-framelen_dec/2 : framelen_dec/2-1);
v = 1:chirplen_dec;
x = (u-1-framelen/2) * 0.0156;
y = (v-1) * 0.04;
c = fftshift(fft2(b), 2);
scans = abs(c);

if doplot
    f = waitbar(0, 'Plotting frames');
    for i = 1:numframes
        imcomplex(x, y, abs(c(v, u, i)));
        f = waitbar(i/numframes, f, 'Plotting frames');
        pause(duration / numframes);
    end
    close(f);
end

timestamps = posixtime(dt_start) : duration/numframes : posixtime(dt_end);

end
