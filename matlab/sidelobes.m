N = 256;
w=linspace(-pi, pi, 10*N);
a = ones(N,1);
n=0:N-1;
b=exp(-1j*n.'*w).'*a;
f =w/pi;
figure(1); plot((f), (abs(b))/N)
hold on;
xline([1/N,-1/N],'r--')
figure(2); plot((f), 20*log10(abs(b)/N));
hold on;
xline([1/N,-1/N],'r--')
