classdef OccupancyMap3d
    
    properties
        x;
        y;
        z;
        v;
        cx;
        cy;
        cz;
    end
    
    methods
        function obj = OccupancyMap3d(sx, sy, sz, rx, ry, rz, cx, cy, cz)
            lsx = cx + (-sx/2+sx/rx/2 : sx/rx : sx/2-sx/rx/2);
            lsy = cy + (-sy/2+sy/ry/2 : sy/ry : sy/2-sy/ry/2);
            lsz = cz + (-sz/2+sz/rz/2 : sz/rz : sz/2-sz/rz/2);
            [obj.y, obj.x, obj.z] = meshgrid(lsy, lsx, lsz);
            obj.v = zeros(rx, ry, rz, 'double');
            obj.cx = cx;
            obj.cy = cy;
            obj.cz = cz;
        end
        
        function [u, v, w] = xyz2uvw_clipped(obj, x, y, z)
            [resx, resy, resz] = size(obj.v);
            ppmx = 1 / (obj.x(2, 1, 1) - obj.x(1, 1, 1));
            ppmy = 1 / (obj.y(1, 2, 1) - obj.y(1, 1, 1));
            ppmz = 1 / (obj.z(1, 1, 2) - obj.z(1, 1, 1));
            u = min(max(round((x-obj.cx)*ppmx + resx/2 + 0.5), 1), resx);
            v = min(max(round((y-obj.cy)*ppmy + resy/2 + 0.5), 1), resy);
            w = min(max(round((z-obj.cz)*ppmz + resz/2 + 0.5), 1), resz);
        end

        function obj = render_cube(obj, x, y, z, sx, sy, sz, v)
            [umin, vmin, wmin] = obj.xyz2uvw_clipped(x-sx/2, y-sy/2, z-sz/2);
            [umax, vmax, wmax] = obj.xyz2uvw_clipped(x+sx/2, y+sy/2, z+sz/2);
            obj.v(umin:umax, vmin:vmax, wmin:wmax) = v;
        end
    end
end

