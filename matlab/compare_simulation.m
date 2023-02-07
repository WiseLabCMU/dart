datadir = 'D:\dartdata';
dataset = 'cubes';

real_rad = rad;
load(simfile, 'rad');
sim = rad;
rad = real_rad;
clear real_rad;

x = linspace(min_doppler, max_doppler, res_doppler);
y = linspace(min_range, max_range/2, floor(res_range/2));

N = size(rad, 1);
ts = t(2)-t(1);
for i = 1:N
    c = squeeze(rad(i,1:64,:));
    d = squeeze(sim(i,1:64,:));
%     f = squeeze(pred(i,:,:));
    c(:,33) = 0;
    d(:,33) = 0;
%     c = c - min(c, [], 2);
%     c = c ./ max(c, [], 2);
%     d = d - min(d, [], 2);
%     d = d ./ max(d, [], 2);
    subplot(2,1,1);
    imcomplex(x, y, c);
    title('real');
    subplot(2,1,2);
    imcomplex(x, y, d);
    title('sim');
%     subplot(3,1,3);
%     imcomplex(x, y, f);
%     title('pred');
    pause(ts);
end
