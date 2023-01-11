% function scans = scans_from_file(filename)
filename = 'E:\dartdata\CupData\cup.mat';

load(filename, 'frames', 'start_time', 'end_time');
dt_start = datetime(start_time, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS''Z''');
dt_end = datetime(end_time, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS''Z''');
duration = posixtime(dt_end) - posixtime(dt_start);

framelen = 256;
chirplen = size(frames, 4);
numframes = floor(size(frames, 1) / framelen);
numsamples = numframes * framelen;
dt = duration / numframes;

a = squeeze(frames(1:numsamples, 1, 1, :));
b = reshape(a.', chirplen, framelen, []);

x = 0:framelen-1;
y = 0:chirplen-1;
f = waitbar(0, 'Plotting frames');
for i = 1:numframes
    c = fftshift(fft2(b(:,:,i)), 2);
    plotComplex(x, y, c);
    f = waitbar(i/numframes, f, 'Plotting frames');
    pause(dt);
end

% end
