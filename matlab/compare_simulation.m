close all;

datadir = 'D:\dartdata';
dataset = 'cubes';

real_rad = rad;
load(simfile, 'rad');
sim_rad = rad;
rad = real_rad;

x = linspace(min_doppler, max_doppler, res_doppler);
y = linspace(min_range, max_range/2, floor(res_range/2));

fig = figure;
subplot(2,3,[2,3,5,6]);
facing = zeros(size(pos));
for i = 1:size(facing, 1)
    r = squeeze(rot(i, :, :));
    facing(i, :) = (r * [0.3; 0; 0]).';
end
v = max(min(vel,0.4),-0.4);
[fo, vo] = isosurface(map.x,map.y,map.z,map.v);
patch('Faces',fo,'Vertices',vo,'FaceColor','#909090','EdgeColor','None');
hold on;
light;
cam = plotCamera('Size',0.1,'Opacity',0.1);
axis equal; axis vis3d;
axis([-2,3,-2,3,0,2]);

N = size(rad, 1);
sld = uicontrol('Parent',fig,'Style','slider','Value',1,'Min',1,'Max',N,'Position',[1200,125,1200,50],'SliderStep',[1,1]./(N-1));
bl1 = uicontrol('Parent',fig,'Style','text','Position',[1150,135,23,30],'String','0','FontSize',16);
tend = string(t(end)-t(1));
bl2 = uicontrol('Parent',fig,'Style','text','Position',[2420,135,100,30],'String',tend,'FontSize',16);
bl3 = uicontrol('Parent',fig,'Style','text','Position',[1750,90,100,30],'String','Time (s)','FontSize',16);
btn = uicontrol('Parent',fig,'Style','checkbox','Position',[1250,190,100,30],'String','Auto','Value',false,'FontSize',16);

q1 = quiver3(0,0,0,0,0,0,'k', ...
    'XDataSource','pos(sld.Value,1)', ...
    'YDataSource','pos(sld.Value,2)', ...
    'ZDataSource','pos(sld.Value,3)', ...
    'UDataSource','v(sld.Value,1)', ...
    'VDataSource','v(sld.Value,2)', ...
    'WDataSource','v(sld.Value,3)', ...
    'LineWidth',2);
q2 = quiver3(0,0,0,0,0,0, ...
    'XDataSource','pos(sld.Value,1)', ...
    'YDataSource','pos(sld.Value,2)', ...
    'ZDataSource','pos(sld.Value,3)', ...
    'UDataSource','facing(sld.Value,1)', ...
    'VDataSource','facing(sld.Value,2)', ...
    'WDataSource','facing(sld.Value,3)', ...
    'LineWidth',2);


ts = t(2)-t(1);
while true
    if btn.Value
        sld.Value = min(sld.Value + 1, sld.Max);
    end
    sld.Value = round(sld.Value);
    c = squeeze(real_rad(sld.Value,1:64,:));
    d = squeeze(sim_rad(sld.Value,1:64,:));
    c(:,33) = 0;
    d(:,33) = 0;
    subplot(2,3,1);
    imcomplex(x, y, c);
    subplot(2,3,4);
    imcomplex(x, y, d);
    r = squeeze(rot(sld.Value,:,:)) * axang2rotm([0,1,0,deg2rad(90)]);
    p = pos(sld.Value,:);
    pose = rigidtform3d(r, p);
    cam.AbsolutePose = pose;
    refreshdata(q1, 'caller');
    refreshdata(q2, 'caller');
    pause(ts);
end
