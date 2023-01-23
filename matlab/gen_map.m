function map = gen_map()

sz = [1.53, 1.71, 0.84];
res = sz * 100;
center = [0.665, 0.545, 0.38];

s = [0.38, 0.45, 0.64];
c = [[0.19, 0.015, 0.38]; ...
          [0.19, 1.075, 0.38]; ...
          [1.14, 0.565, 0.38]];
v = 1;

map = OccupancyMap3d(sz(1), sz(2), sz(3), res(1), res(2), res(3), center(1), center(2), center(3));
for i = 1:size(c, 1)
    map = map.render_cube(c(i,1), c(i,2), c(i,3), s(1), s(2), s(3), v);
end

end

