classdef OccupancyMap3d
    
    properties
        x;
        y;
        z;
        v;
    end
    
    methods
        function obj = OccupancyMap3d(sz, res)
            ls = -sz/2+sz/res/2 : sz/res : sz/2-sz/res/2;
            [obj.y, obj.x, obj.z] = meshgrid(ls, ls, ls);
            obj.v = zeros(res, res, res, 'double');
        end
        
        function [u, v, w] = xyz2uvw_clipped(obj, x, y, z)
            [resx, resy, resz] = size(obj.v);
            ppmx = 1 / (obj.x(2, 1, 1) - obj.x(1, 1, 1));
            ppmy = 1 / (obj.y(1, 2, 1) - obj.y(1, 1, 1));
            ppmz = 1 / (obj.z(1, 1, 2) - obj.z(1, 1, 1));
            u = min(max(round(x*ppmx + resx/2 + 0.5), 1), resx);
            v = min(max(round(y*ppmy + resy/2 + 0.5), 1), resy);
            w = min(max(round(z*ppmz + resz/2 + 0.5), 1), resz);
        end

        function obj = render_cube(obj, x, y, z, s, v)
            [umin, vmin, wmin] = obj.xyz2uvw_clipped(x-s/2, y-s/2, z-s/2);
            [umax, vmax, wmax] = obj.xyz2uvw_clipped(x+s/2, y+s/2, z+s/2);
            obj.v(umin:umax, vmin:vmax, wmin:wmax) = v;
        end
    end
end

