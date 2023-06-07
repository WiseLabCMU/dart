function map = gen_map_linear()

sz = [2.00, 2.00, 1.00];
res = sz * 100;
center = [0.00, 0.00, 0.00];

s = [0.01, 0.30, 1.00];
c = [0.00, 0.00, 0.00];
v = 1;

map = OccupancyMap3d(sz(1), sz(2), sz(3), res(1), res(2), res(3), center(1), center(2), center(3));
for i = 1:size(c, 1)
    map = map.render_cube(c(i,1), c(i,2), c(i,3), s(1), s(2), s(3), v);
end

end

