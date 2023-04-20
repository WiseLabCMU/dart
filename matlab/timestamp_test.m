DATAPATH = 'D:/dartdata/cabinets-sep';
NUM_TRACES = 20;

sss = [];
eee = [];
ttt = [];
rrr = [];
for i = 0 : NUM_TRACES - 1
    dataset = sprintf('cabinets-%03d', i);
    data_from_file;
%     scanfile = fullfile(DATAPATH, dataset, 'frames', dataset + ".mat");
%     trajfile = fullfile(DATAPATH, dataset, 'traj', dataset + ".mat");
%     load(scanfile, 'start_time', 'end_time');
%     load(trajfile, 't', 'rt');
%     sss = [sss; posixtime(datetime(start_time, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS''Z''')) - 5 * 60 * 60];
%     eee = [eee; posixtime(datetime(end_time, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS''Z''')) - 5 * 60 * 60];
%     ttt = [ttt; t];
%     rrr = [rrr; rt];
end
