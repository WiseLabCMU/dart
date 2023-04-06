close all;

datadir = 'F:\dartdata';
dataset = 'linear2';

real_rad = rad;
% load(simfile, 'rad');
% sim_rad = rad;
rad = real_rad;

x = linspace(min_doppler, max_doppler, res_doppler);
y = linspace(min_range, max_range, res_range);
% x = [min_doppler, max_doppler];
% y = [min_range, max_range];

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
% axis([-2,3,-2,3,0,2]);
axis([-3,3,-3,3,-1.5,1.5]);
view(3);
xlabel('x (m)');
ylabel('y (m)');
zlabel('z (m)');
title('Scene');
set(gca, 'FontSize', 16);

N = size(rad, 1);
sld = uicontrol('Parent',fig,'Style','slider','Value',1,'Min',1,'Max',N,'Position',[1200,125,1200,50],'SliderStep',[1,1]./(N-1));
bl1 = uicontrol('Parent',fig,'Style','text','Position',[1150,135,23,30],'String','0','FontSize',16);
tend = string(t(end)-t(1));
bl2 = uicontrol('Parent',fig,'Style','text','Position',[2420,135,100,30],'String',tend,'FontSize',16);
bl3 = uicontrol('Parent',fig,'Style','text','Position',[1750,90,100,30],'String','Time (s)','FontSize',16);
btn = uicontrol('Parent',fig,'Style','checkbox','Position',[1250,90,100,30],'String','Auto','Value',false,'FontSize',16);

global sel_plots;
sel_plots = dictionary;

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

fig.WindowState = 'maximized';
ts = t(2)-t(1);
last_idx = 0;
while true
    if btn.Value
        sld.Value = min(sld.Value + 1, sld.Max);
    end
    sld.Value = round(sld.Value);
    if sld.Value ~= last_idx
        last_idx = sld.Value;
        if sel_plots.numEntries() > 0
            for k = sel_plots.keys()
                delete(sel_plots(k));
                sel_plots(k) = [];
            end
        end

        rr = squeeze(rot(sld.Value,:,:));
        pp = pos(sld.Value,:);
        vv = vel(sld.Value,:);
        pose = rigidtform3d(rr * axang2rotm([0,1,0,deg2rad(90)]), pp);
        cam.AbsolutePose = pose;
        refreshdata(q1, 'caller');
        refreshdata(q2, 'caller');

        subplot(2,3,1);
        hold off;
%         c = fliplr(squeeze(real_rad(sld.Value,1:64,:)));
        c = squeeze(real_rad(sld.Value,:,:));
%         c(:,32) = 0;
        image('XData', x, 'YData', y, 'CData', (c - min(c(:))) / (max(c(:)) - min(c(:))) * 255, 'ButtonDownFcn', {@pixelclick_callback,pp,vv,rr});
        axis tight;
        xlabel('Doppler (m/s)');
        ylabel('Range (m)');
        title('Real Scans');
        set(gca, 'FontSize', 16);
    
        subplot(2,3,4);
        hold off;
%         d = fliplr(squeeze(sim_rad(sld.Value,1:64,:)));
        d = squeeze(real_rad(sld.Value,:,:));
%         d(:,32) = 0;
        image(x, y, (d - min(d(:))) / (max(d(:)) - min(d(:))) * 255, 'ButtonDownFcn', {@pixelclick_callback,pp,vv,rr});
        axis tight; axis xy;
        xlabel('Doppler (m/s)');
        ylabel('Range (m)');
        title('Simulated Scans');
        set(gca, 'FontSize', 16);
    end
    drawnow;
end

function pixelclick_callback(src, event, pos, vel, rot)
global sel_plots;

d = interp1(src.XData,src.XData,event.IntersectionPoint(1),'nearest');
r = interp1(src.YData,src.YData,event.IntersectionPoint(2),'nearest');
key = struct;
key.d = d;
key.r = r;
key.p = 1;

if event.Button == 1 && (sel_plots.numEntries == 0 || ~sel_plots.isKey(key))
    s = norm(vel);
    v = rot.' * vel.' / s;
    [~,~,V] = svd(eye(3)-v*v');
    p = V(:,1);
    q = V(:,2);
    dnorm = d/s;
    
    psi = linspace(0,2*pi,256);
    t = r*(sqrt(1-dnorm^2)*(p*cos(psi)+q*sin(psi))+v*dnorm);
    tworld = pos.'+rot*t;
    
    subplot(2,3,1);
    hold on;
    key.p = 1;
    sel_plots(key) = plot(d,r,'rs','LineWidth',2,'MarkerSize',10,'PickableParts','None');
    
    subplot(2,3,4);
    hold on;
    key.p = 2;
    sel_plots(key) = plot(d,r,'rs','LineWidth',2,'MarkerSize',10,'PickableParts','None');
    
    subplot(2,3,[2,3,5,6]);
    key.p = 3;
    sel_plots(key) = plot3(tworld(1,:),tworld(2,:),tworld(3,:),'r','Linewidth',2);
elseif event.Button == 3 && sel_plots.numEntries > 0 && sel_plots.isKey(key)
    key.p = 1;
    delete(sel_plots(key));
    sel_plots(key) = [];
    key.p = 2;
    delete(sel_plots(key));
    sel_plots(key) = [];
    key.p = 3;
    delete(sel_plots(key));
    sel_plots(key) = [];
end

end
