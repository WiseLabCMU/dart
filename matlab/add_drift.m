STD_XYZ_DRIFT = 0.01;   % std drift per m traveled
VAR_XYZ_NOISE = 0.0025; % var iid positional noise per m traveled

dpos = diff(pos);
ddist = vecnorm(dpos, 2, 2);
pdrift = STD_XYZ_DRIFT * randn;
pnoise = sqrt(VAR_XYZ_NOISE .* ddist) .* randn(size(dpos));
pnoisy = cumsum([pos(1, :); dpos + pdrift * ddist + pnoise]);

plot(pos(:, 1), pos(:, 2)); hold on;
plot(pnoisy(:, 1), pnoisy(:, 2)); axis equal;
