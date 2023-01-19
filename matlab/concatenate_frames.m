clear;

files = {
    '../data/frames.mat'
    '../data/frames2.mat'
    '../data/frames3.mat'
    '../data/frames4.mat'
};

all_t = [];
all_rad = [];
all_pos = [];
all_rot = [];
all_vel = [];
for i = 1:length(files)
    load(files{i}, 't', 'rad', 'pos', 'rot', 'vel');
    all_t = [all_t; t];
    all_rad = [all_rad; rad];
    all_pos = [all_pos; pos];
    all_rot = [all_rot; rot];
    all_vel = [all_vel; vel];
end

t = all_t;
rad = all_rad;
pos = all_pos;
rot = all_rot;
vel = all_vel;

save('../data/all_frames', 't', 'rad', 'pos', 'rot', 'vel');
