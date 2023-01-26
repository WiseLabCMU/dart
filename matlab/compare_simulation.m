datadir = 'D:\dartdata';
dataset = 'cubes';

real_rad = rad;
load(simfile, 'rad');
sim = rad;
rad = real_rad;
clear real_rad;

x = linspace(min_doppler, max_doppler, res_doppler);
y = linspace(min_range, max_range, floor(res_range));

N = size(rad, 1);
ts = t(2)-t(1);
for i = 1:N
    c = squeeze(rad(i,:,:));
    d = squeeze(sim(i,:,:));
    c(:,33) = 0;
    subplot(2,1,1);
    imcomplex(x, y, c);
    subplot(2,1,2);
    imcomplex(x, y, d);
    pause(ts);
end
