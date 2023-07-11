r1 = 1;
r2 = 1.04;
rr = 1.02;
r3 = 2;
az = deg2rad(15);
el = deg2rad(50);
N = 100;
vang1 = deg2rad(47);
vang2 = deg2rad(43);
vvang = deg2rad(45);
vdir = deg2rad(20);
[h, psi] = meshgrid(linspace(0,r3,N),linspace(-pi,pi,N));
vvx = h;
vr = h.*tan(vvang/2);
vvy = vr.*cos(psi);
vz = vr.*sin(psi);
vx = vvx.*cos(vdir)-vvy.*sin(vdir);
vy = vvx.*sin(vdir)+vvy.*cos(vdir);

vvxring1 = r1.*cos(vang1/2);
vrring1 = r1.*sin(vang1/2);
vvyring1 = vrring1.*cos(psi);
vzring1 = vrring1.*sin(psi);
vxring1 = vvxring1.*cos(vdir)-vvyring1.*sin(vdir);
vyring1 = vvxring1.*sin(vdir)+vvyring1.*cos(vdir);
vring1_az = atan2(vyring1, vxring1);
vring1_el = atan2(vzring1, hypot(vxring1, vyring1));
vring1_valid = (abs(vring1_az) < az / 2) & (abs(vring1_el) < el / 2);
vxring1_v = vxring1(vring1_valid);
vyring1_v = vyring1(vring1_valid);
vzring1_v = vzring1(vring1_valid);

vvxring2 = r1.*cos(vang2/2);
vrring2 = r1.*sin(vang2/2);
vvyring2 = vrring2.*cos(psi);
vzring2 = vrring2.*sin(psi);
vxring2 = vvxring2.*cos(vdir)-vvyring2.*sin(vdir);
vyring2 = vvxring2.*sin(vdir)+vvyring2.*cos(vdir);
vring2_az = atan2(vyring2, vxring2);
vring2_el = atan2(vzring2, hypot(vxring2, vyring2));
vring2_valid = (abs(vring2_az) < az / 2) & (abs(vring2_el) < el / 2);
vxring2_v = vxring2(vring2_valid);
vyring2_v = vyring2(vring2_valid);
vzring2_v = vzring2(vring2_valid);

vvxring3 = r2.*cos(vang1/2);
vrring3 = r2.*sin(vang1/2);
vvyring3 = vrring3.*cos(psi);
vzring3 = vrring3.*sin(psi);
vxring3 = vvxring3.*cos(vdir)-vvyring3.*sin(vdir);
vyring3 = vvxring3.*sin(vdir)+vvyring3.*cos(vdir);
vring3_az = atan2(vyring3, vxring3);
vring3_el = atan2(vzring3, hypot(vxring3, vyring3));
vring3_valid = (abs(vring3_az) < az / 2) & (abs(vring3_el) < el / 2);
vxring3_v = vxring3(vring3_valid);
vyring3_v = vyring3(vring3_valid);
vzring3_v = vzring3(vring3_valid);

vvxring4 = r2.*cos(vang2/2);
vrring4 = r2.*sin(vang2/2);
vvyring4 = vrring4.*cos(psi);
vzring4 = vrring4.*sin(psi);
vxring4 = vvxring4.*cos(vdir)-vvyring4.*sin(vdir);
vyring4 = vvxring4.*sin(vdir)+vvyring4.*cos(vdir);
vring4_az = atan2(vyring4, vxring4);
vring4_el = atan2(vzring4, hypot(vxring4, vyring4));
vring4_valid = (abs(vring4_az) < az / 2) & (abs(vring4_el) < el / 2);
vxring4_v = vxring4(vring4_valid);
vyring4_v = vyring4(vring4_valid);
vzring4_v = vzring4(vring4_valid);

pr1_x = [vxring1;flipud(vxring3)];
pr1_y = [vyring1;flipud(vyring3)];
pr1_z = [vzring1;flipud(vzring3)];
pr1_xv = [vxring1_v;flipud(vxring3_v)];
pr1_yv = [vyring1_v;flipud(vyring3_v)];
pr1_zv = [vzring1_v;flipud(vzring3_v)];

pr2_x = [vxring2;flipud(vxring4)];
pr2_y = [vyring2;flipud(vyring4)];
pr2_z = [vzring2;flipud(vzring4)];
pr2_xv = [vxring2_v;flipud(vxring4_v)];
pr2_yv = [vyring2_v;flipud(vyring4_v)];
pr2_zv = [vzring2_v;flipud(vzring4_v)];

pr3_x = [vxring1;flipud(vxring2)];
pr3_y = [vyring1;flipud(vyring2)];
pr3_z = [vzring1;flipud(vzring2)];
pr3_xv = [vxring1_v;flipud(vxring2_v)];
pr3_yv = [vyring1_v;flipud(vyring2_v)];
pr3_zv = [vzring1_v;flipud(vzring2_v)];

pr4_x = [vxring3;flipud(vxring4)];
pr4_y = [vyring3;flipud(vyring4)];
pr4_z = [vzring3;flipud(vzring4)];
pr4_xv = [vxring3_v;flipud(vxring4_v)];
pr4_yv = [vyring3_v;flipud(vyring4_v)];
pr4_zv = [vzring3_v;flipud(vzring4_v)];

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
mesh(rr.*x,rr.*y,rr.*z,'EdgeColor','none','FaceColor','r','FaceAlpha',0.2);
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
% mesh(vx,vy,vz,'EdgeColor','none','FaceColor','g','FaceAlpha',0.2);
% patch(pr1_x,pr1_y,pr1_z,'y','FaceAlpha',1,'EdgeColor','k');
% patch(pr2_x,pr2_y,pr2_z,'y','FaceAlpha',1,'EdgeColor','k');
% patch(pr3_x,pr3_y,pr3_z,'y','FaceAlpha',1,'EdgeColor','k');
% patch(pr4_x,pr4_y,pr4_z,'y','FaceAlpha',1,'EdgeColor','k');
% patch(pr1_xv,pr1_yv,pr1_zv,'k','FaceAlpha',1,'EdgeColor','k');
% patch(pr2_xv,pr2_yv,pr2_zv,'k','FaceAlpha',1,'EdgeColor','k');
% patch(pr3_xv,pr3_yv,pr3_zv,'k','FaceAlpha',1,'EdgeColor','k');
% patch(pr4_xv,pr4_yv,pr4_zv,'k','FaceAlpha',1,'EdgeColor','k');
axis equal;axis vis3d;
set(gca,'XTick',[],'YTick',[],'ZTick',[]);
axis off;

