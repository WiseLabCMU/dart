function varargout = plotComplex(x, y, val, varargin)

a = abs(val).^.5;

h = (angle(val) + pi) / (2*pi);
s = (a - min(a(:))) / (max(a(:)) - min(a(:)));
v = ones(size(h));

hsv = cat(3, h, s, v);
rgb = hsv2rgb(hsv);
[ind, map] = rgb2ind(rgb, 65536, 'nodither');

[varargout{1 : nargout}] = pcolor(x, y, ind, varargin{:});
% [varargout{1 : nargout}] = scatter(x(:), y(:), 'filled', 'CData', ind(:), varargin{:});
shading flat;
axis equal tight;
colormap(map);

end

