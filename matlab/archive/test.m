[theta, phi] = meshgrid(linspace(-pi/2,pi/2,100),linspace(-pi/2,pi/2,100));

theta_ = theta / pi * 180 / 56;
phi_ = phi / pi * 180 / 56;

gain = exp(((0.14*theta_.^6+0.13*theta_.^4-8.2*theta_.^2)+(3.1*phi_.^8-22*phi_.^6+54*phi_.^4-55*phi_.^2))/10);