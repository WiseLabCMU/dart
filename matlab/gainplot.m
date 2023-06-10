[theta, phi] = meshgrid(linspace(-pi/2,pi/2,100),linspace(-pi/2,pi/2,100));

x = cos(phi).*cos(theta);
y = cos(phi).*sin(theta);
z = sin(phi);

theta_ = theta / pi * 180 / 56;
phi_ = phi / pi * 180 / 56;

gaindb = ((0.14*theta_.^6+0.13*theta_.^4-8.2*theta_.^2)+(3.1*phi_.^8-22*phi_.^6+54*phi_.^4-55*phi_.^2));
gain = 10.^(gaindb/20);

figure;
mesh(gain.*x,gain.*y,gain.*z,'CData',gain)
axis equal;axis vis3d
xlabel('x');ylabel('y');zlabel('z');
title('Base Gain');

N = 8;
w = sin(theta)*pi;
a = ones(N,1);
n = 0:N-1;

for b = 0:N-1
    bin = (b/N * 2 - 1)*pi;
    arrayfactor = reshape(abs(exp(-1j*n.'*(w(:).'-bin)).'*a)/N, size(w));
    figure;
    mesh(gain.*arrayfactor.*x,gain.*arrayfactor.*y,gain.*arrayfactor.*z,'CData',gain.*arrayfactor)
    axis equal;axis vis3d;
    view(2);
    xlim([0,1]);
    ylim([-1,1]);
    zlim([-1,1]);
    xlabel('x');ylabel('y');zlabel('z');
    title(sprintf('Bin %d', b));
end
