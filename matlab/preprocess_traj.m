function preprocess_traj(infile, outfile, fs)

s = readlines(infile);
N = size(s, 1) - 1; % Skip last line to handle EOF
fprintf('Loading %s...\n', infile);

pose = jsondecode(s(1)).data;
dt = datetime(pose.ts_at_receive, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS''Z''');
last_t = posixtime(dt);
last_x = pose.position.x;
last_y = pose.position.y;
last_z = pose.position.z;
tt = last_t;
for i = 2:N
    pose = jsondecode(s(i)).data;
    dt = datetime(pose.ts_at_receive, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS''Z''');
    cur_t = posixtime(dt);
    cur_x = pose.position.x;
    cur_y = pose.position.y;
    cur_z = pose.position.z;
    while tt < cur_t
        pct = (tt - last_t) / (cur_t - last_t);
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
    last_t = cur_t;
    last_x = cur_x;
    last_y = cur_y;
    last_z = cur_z;
end

end
