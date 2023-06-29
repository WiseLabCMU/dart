[theta, phi] = meshgrid(linspace(-pi/2,pi/2,100),linspace(-pi/2,pi/2,100));
% theta = linspace(-pi/2,pi/2,100);
% phi = 0;

y = cos(phi).*cos(theta);
x = cos(phi).*sin(theta);
z = sin(phi);

theta_ = theta / pi * 180 / 56;
phi_ = phi / pi * 180 / 56;

gaindb = ((0.14*theta_.^6+0.13*theta_.^4-8.2*theta_.^2)+(3.1*phi_.^8-22*phi_.^6+54*phi_.^4-55*phi_.^2));
gain = 10.^(gaindb/20);

% figure;
% mesh(gain.*x,gain.*y,gain.*z,'CData',gain)
% axis equal;axis vis3d
% xlabel('x');ylabel('y');zlabel('z');
% title('Base Gain');

N = 8;
w = sin(theta)*pi;
a = ones(N,1);
n = 0:N-1;

figure;
t = tiledlayout(1,8,'TileSpacing','Tight','Padding','Tight');
for b = 0:N-1
    nexttile;
    bin = (b/N * 2 - 1)*pi;
    arrayfactor = reshape(abs(exp(-1j*n.'*(w(:).'-bin)).'*a)/N, size(w));
    % subplot(1,8,b+1);
    mesh(gain.*arrayfactor.*x,gain.*arrayfactor.*y,gain.*arrayfactor.*z,'CData',gain.*arrayfactor)
    % axis equal;axis vis3d;
    % plot(gain.*arrayfactor.*x, gain.*arrayfactor.*y);
    axis equal;
    set(gca,'XTick',[],'YTick',[],'ZTick',[]);
    axis off;
    view(2);
    ylim([0,1]);
    xlim([-0.4,0.4]);
    zlim([-1,1]);
    % xlabel('x');ylabel('y');zlabel('z');
    % title(sprintf('Bin %d', b));
end
