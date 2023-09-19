sigma = h5read('~/dart/results/cichallway_fixed/map.h5', '/sigma');
alpha = h5read('~/dart/results/cichallway_fixed/map.h5', '/alpha');
s = sigma;%(75:225,306:606,300:450);
a = alpha;%(75:225,306:606,300:450);
sss = (s-min(s(:)))./(max(s(:))-min(s(:)));
aaa = (a-min(a(:)))./(max(a(:))-min(a(:)));

vvv = volshow(1-10.^(-s));