% function scans = scans_from_file(filename)
filename = 'D:\CupData\cup.mat';

if ~exist('frames', 'var')
    load(filename, 'frames');
end
if ~exist('start_time', 'var')
    load(filename, 'start_time');
end
if ~exist('end_time', 'var')
    load(filename, 'end_time');
end
dt_start = datetime(start_time, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS''Z''');
dt_end = datetime(end_time, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS''Z''');
duration = posixtime(dt_end) - posixtime(dt_start);

range_decimation = 16;
doppler_decimation = 8;
framelen = 1024; % before decimation
chirplen = size(frames, 4); % before decmiation
numframes = floor(size(frames, 1) / framelen);
numsamples = numframes * framelen;
dt = duration / numframes;

a = squeeze(frames(1:numsamples, 1, 1, :));
b = reshape(a.', chirplen, framelen, []);

framelen_dec = framelen / doppler_decimation;
chirplen_dec = chirplen / range_decimation;
u = framelen/2 + (-framelen_dec/2 : framelen_dec/2-1);
v = 1:chirplen_dec;
x = (u-1-framelen/2) * 0.0156;
y = (v-1) * 0.04;
c = fftshift(fft2(b), 2);
f = waitbar(0, 'Plotting frames');
for i = 1:numframes
    plotComplex(x, y, c(v, u, i));
    f = waitbar(i/numframes, f, 'Plotting frames');
    pause(dt);
end
close(f);

% end
