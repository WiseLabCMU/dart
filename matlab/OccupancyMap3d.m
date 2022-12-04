classdef OccupancyMap3d
    
    properties
        x;
        y;
        z;
        v;
    end
    
    methods
        function obj = OccupancyMap3d(sz, res)
            ls = linspace(-sz/2, sz/2, res);
            [obj.y, obj.x, obj.z] = meshgrid(ls, ls, ls);
            obj.v = zeros(res, res, res, 'double');
        end
        
        function [u, v, w] = xyz2uvw_clipped(obj, x, y, z)
            [resx, resy, resz] = size(obj.v);
            ppmx = resx / (obj.x(end, end, end) - obj.x(1, 1, 1));
            ppmy = resy / (obj.y(end, end, end) - obj.y(1, 1, 1));
            ppmz = resz / (obj.z(end, end, end) - obj.z(1, 1, 1));
            u = min(max(round(x*ppmx + resx/2), 1), resx);
            v = min(max(round(y*ppmy + resy/2), 1), resy);
            w = min(max(round(z*ppmz + resz/2), 1), resz);
        end

        function obj = render_cube(obj, x, y, z, s, v)
            [umin, vmin, wmin] = obj.xyz2uvw_clipped(x-s/2, y-s/2, z-s/2);
            [umax, vmax, wmax] = obj.xyz2uvw_clipped(x+s/2, y+s/2, z+s/2);
            obj.v(umin:umax, vmin:vmax, wmin:wmax) = v;
        end
    end
end

