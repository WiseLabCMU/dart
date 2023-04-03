function preprocess_traj(infile, outfile, fs)

s = readlines(infile);
N = size(s, 1) - 1; % Skip last line to handle EOF
fprintf('Loading %s...\n', infile);

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

tt = t(1);
for i = 2:N
    pose = jsondecode(s(i)).data;
    dt = datetime(pose.ts_at_receive, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS''Z''');
    t(i) = posixtime(dt);
    cur_x = pose.position.x;
    cur_y = pose.position.y;
    cur_z = pose.position.z;
    while tt < t(i)
        pct = (tt - t(i - 1)) / (t(i) - t(i - 1));
        trajjson = struct();
        trajjson.object_id = 'mmwave-radar';
        trajjson.data = struct();
        trajjson.data.position = struct();
        trajjson.data.position.x = pct * (cur_x - last_x);
        trajjson.data.position.y = pct * (cur_y - last_y);
        trajjson.data.position.z = pct * (cur_z - last_z);
        trajjson.data.rotation = struct();
        trajjson.data.rotation.x = 0.0;
        trajjson.data.rotation.y = 0.0;
        trajjson.data.rotation.z = 0.0;
        trajjson.data.rotation.w = 1.0;
        trajjson.data.timestamp = tt;
        dt = datetime(tt, 'ConvertFrom', 'posixtime', 'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS''Z''');
        trajjson.data.ts_at_receive = dt;
        jsonstring = jsonencode(trajjson);
        writelines(jsonstring, outfile, 'WriteMode', 'append');
        tt = tt + 1 / fs;
    end
    last_x = cur_x;
    last_y = cur_y;
    last_z = cur_z;
end

end
