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

chirplen = size(frames, 4); % before decmiation
numframes = floor(size(frames, 1) / framelen);
numsamples = numframes * framelen;

t_start = posixtime(datetime(start_time, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS''Z'''));
t_start = t_start - 5*60*60;
t_end = posixtime(datetime(end_time, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS''Z'''));
t_end = t_end - 5*60*60;
ts = (t_end - t_start) / numframes;

timestamps = (t_start : ts : t_end-ts).';

a = squeeze(frames(1:numsamples, 1, 1, :));
b = reshape(a.', chirplen, framelen, []);

framelen_dec = framelen / doppler_decimation;
chirplen_dec = chirplen / range_decimation;
u = framelen/2 + (-framelen_dec/2 : framelen_dec/2-1);
v = 1:chirplen_dec;
x = (u-1-framelen/2) * 0.0156;
y = (v-1) * 0.04;
c = permute(fftshift(fft2(b), 2), [3 1 2]);
scans = abs(c);

if doplot
    f = waitbar(0, 'Plotting frames');
    for i = 1:numframes
        imcomplex(x, y, abs(c(i, v, u)));
        f = waitbar(i/numframes, f, 'Plotting frames');
        pause(ts);
    end
    close(f);
end


end
