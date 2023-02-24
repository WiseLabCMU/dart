function varargout = imcomplex(x, y, img, varargin)

% a = abs(img).^.5;
a = abs(img);

h = (angle(img) + pi) / (2*pi);
s = ones(size(h));
v = (a - min(a(:))) / (max(a(:)) - min(a(:)));

hsv = cat(3, h, s, v);
rgb = hsv2rgb(hsv);

[varargout{1 : nargout}] = image(x, y, rgb, varargin{:});
axis tight xy;

end

