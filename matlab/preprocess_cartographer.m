function preprocess_cartographer(infile, outfile)

s = readmatrix(infile);
N = size(s, 1) - 1; % Skip last line to handle EOF
fprintf('Converting %s...\n', infile);

rt = s(1:N-1, 1) * 1e-9;
t = s(1:N-1, 3) * 1e-9;
x = s(1:N-1, 6);
y = s(1:N-1, 7);
z = s(1:N-1, 8);
qx = s(1:N-1, 9);
qy = s(1:N-1, 10);
qz = s(1:N-1, 11);
qw = s(1:N-1, 12);

save(outfile, 'rt', 't', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', '-v7.3');

end
