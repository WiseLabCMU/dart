"""Verify that all dataset files are present and not out of date."""

import os

base = "data"


def _list_contents(path, files, dataset):
    fmt = []
    for f, deps in files.items():
        f = f.replace("*", dataset)
        if os.path.exists(os.path.join(path, f)):
            mt = os.path.getmtime(os.path.join(path, f))
            mt_deps = [
                os.path.getmtime(os.path.join(path, d)) < mt + 1 for d in deps]
            if all(mt_deps):
                fmt += "x    "
            else:
                fmt += "!    "
        else:
            fmt += "     "
    print("{:<16}{}".format(dataset, ''.join(fmt)))


files = {
    "*_speed_report.pdf": ["trajectory.h5"],
    "data.h5": ["trajectory.h5", "radar.h5"],
    "lidar.bag.pbstream": [],
    "lidar.bag_points.ply": [],
    "map.npz": ["lidar.bag_points.ply"],
    "metadata.json": [],
    "pose.bag": [],
    "radar.h5": ["radarpackets.h5"],
    "radarpackets.h5": [],
    "simulated.h5": ["trajectory.h5"],
    "trajectory.csv": [],
    "trajectory.h5": ["trajectory.csv", "metadata.json"],
    "*.MOV": [],
}

print(
    " " * 16 + "spdf data .pbs .ply mapz meta pose radr pckt sim  tcsv traj"
    "vid")
for dataset in sorted(os.listdir(base)):
    _dataset = os.path.join(base, dataset)
    if not dataset.startswith("_") and os.path.isdir(_dataset):
        _list_contents(_dataset, files, dataset)
