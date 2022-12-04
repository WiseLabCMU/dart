sz = 5;
res = 50;

cube_s = 1;
cube_v = 1;

map = OccupancyMap3d(sz, res);
for xc = linspace(-sz/2+cube_s/2, sz/2-cube_s/2, 3)
    for yc = linspace(-sz/2+cube_s/2, sz/2-cube_s/2, 3)
        for zc = linspace(-sz/2+cube_s/2, sz/2-cube_s/2, 3)
            map = map.render_cube(xc, yc, zc, cube_s, cube_v);
        end
    end
end

volshow(map.v);
