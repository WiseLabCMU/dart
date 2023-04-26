function preprocess_traj(infile, outfile)

s = readlines(infile);
N = size(s, 1) - 1; % Skip last line to handle EOF
fprintf('Converting %s...\n', infile);

rt = zeros(N, 1);
t = zeros(N, 1);
x = zeros(N, 1);
y = zeros(N, 1);
z = zeros(N, 1);
qx = zeros(N, 1);
qy = zeros(N, 1);
qz = zeros(N, 1);
qw = zeros(N, 1);

for i = 1:N
    pose = jsondecode(s(i)).data;
    dt = datetime(pose.ts_at_receive, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS''Z''');
    rt(i) = posixtime(dt);
    t(i) = pose.timestamp;
    x(i) = pose.position.x;
    y(i) = pose.position.y;
    z(i) = pose.position.z;
    qx(i) = pose.rotation.x;
    qy(i) = pose.rotation.y;
    qz(i) = pose.rotation.z;
    qw(i) = pose.rotation.w;
end

save(outfile, 'rt', 't', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', '-v7.3');

end
