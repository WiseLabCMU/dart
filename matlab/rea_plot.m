r1 = 5;
r2 = 5.04;
rr = 5.02;
r3 = 7;
az = deg2rad(15);
el = deg2rad(50);
N = 100;

[theta, phi] = meshgrid(linspace(-pi/2,pi/2,N),linspace(-pi/2,pi/2,N));
[theta_reg, phi_reg] = meshgrid(linspace(-az/2,az/2,floor(N*az/pi)),linspace(-el/2,el/2,floor(N*el/pi)));

x = cos(phi).*cos(theta);
y = cos(phi).*sin(theta);
z = sin(phi);

x_reg = cos(phi_reg).*cos(theta_reg);
y_reg = cos(phi_reg).*sin(theta_reg);
z_reg = sin(phi_reg);

tri_upper_x = [0; r3; r3];
tri_upper_y = [0; r3*tan(az/2); -r3*tan(az/2)];
tri_upper_z = [0; r3*tan(el/2); r3*tan(el/2)];

tri_left_x = [0; r3; r3];
tri_left_y = [0; r3*tan(az/2); r3*tan(az/2)];
tri_left_z = [0; r3*tan(el/2); -r3*tan(el/2)];

p1_x = [r1*x_reg(:,1); flipud(r2*x_reg(:,1))];
p1_y = [r1*y_reg(:,1); flipud(r2*y_reg(:,1))];
p1_z = [r1*z_reg(:,1); flipud(r2*z_reg(:,1))];

p2_x = [r1*x_reg(:,end); flipud(r2*x_reg(:,end))];
p2_y = [r1*y_reg(:,end); flipud(r2*y_reg(:,end))];
p2_z = [r1*z_reg(:,end); flipud(r2*z_reg(:,end))];

p3_x = [r1*x_reg(1,:).'; flipud(r2*x_reg(1,:).')];
p3_y = [r1*y_reg(1,:).'; flipud(r2*y_reg(1,:).')];
p3_z = [r1*z_reg(1,:).'; flipud(r2*z_reg(1,:).')];

p4_x = [r1*x_reg(end,:).'; flipud(r2*x_reg(end,:).')];
p4_y = [r1*y_reg(end,:).'; flipud(r2*y_reg(end,:).')];
p4_z = [r1*z_reg(end,:).'; flipud(r2*z_reg(end,:).')];

figure;
mesh(rr.*x,rr.*y,rr.*z,'EdgeColor','None','FaceColor','r','FaceAlpha',0.2);
hold on;
patch(tri_upper_x,tri_upper_y,tri_upper_z,'b','FaceAlpha',0.1,'EdgeColor','k');
patch(tri_upper_x,tri_upper_y,-tri_upper_z,'b','FaceAlpha',0.1,'EdgeColor','k');
patch(tri_left_x,tri_left_y,tri_left_z,'b','FaceAlpha',0.1,'EdgeColor','k');
patch(tri_left_x,-tri_left_y,tri_left_z,'b','FaceAlpha',0.1,'EdgeColor','k');
mesh(r1*x_reg,r1*y_reg,r1*z_reg,'FaceAlpha',1,'EdgeColor','none','FaceColor','m');
mesh(r2*x_reg,r2*y_reg,r2*z_reg,'FaceAlpha',1,'EdgeColor','none','FaceColor','m');
patch(p1_x,p1_y,p1_z,'m','FaceAlpha',1,'EdgeColor','k');
patch(p2_x,p2_y,p2_z,'m','FaceAlpha',1,'EdgeColor','k');
patch(p3_x,p3_y,p3_z,'m','FaceAlpha',1,'EdgeColor','k');
patch(p4_x,p4_y,p4_z,'m','FaceAlpha',1,'EdgeColor','k');
plot3(r1*x_reg(1,:),r1*y_reg(1,:),r1*z_reg(1,:),'k');
plot3(r2*x_reg(1,:),r2*y_reg(1,:),r2*z_reg(1,:),'k');
plot3(r1*x_reg(end,:),r1*y_reg(end,:),r1*z_reg(end,:),'k');
plot3(r2*x_reg(end,:),r2*y_reg(end,:),r2*z_reg(end,:),'k');
axis equal;axis vis3d;
set(gca,'XTick',[],'YTick',[],'ZTick',[]);
axis off;

