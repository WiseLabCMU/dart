sigma = h5read('~/Desktop/prius.h5', '/sigma');
s = sigma(75:225,306:606,300:450);
sss = (s-min(s(:)))./(max(s(:))-min(s(:)));

vvv = volshow(sss);
