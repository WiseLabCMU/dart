# Data Collection

## Radar Computer
1. Power:
    - power supply charges via USB-C
    - power supply power button
    - hold power supply AC button until AC is shown on screen
2. connect laptop to dart-5 (wiselab2023)
3. AFTER laptop is connected to wifi, plug in NUC + radar and press power on NUC
4. run RDP on laptop.
    - connect to win-qbb0j8pltgl.local
    - username: Administrator
    - password: mmwave76GHz
5. run cmd:
    ```sh
    cd git/dart/scripts
    conda activate dart-datacollect
    python radar.py
    ```
6. wait about 60 seconds until mmwave GUI closes
    - should see "flushing 8192 packets"
    - if radar packets are not showing up (this happens on first boot of NUC):
        1. ctrl-C
        2. open task manager
        3. find `DCA1000EVM` exe
        4. end task
        5. power cycle the radar
        6. start again from the `python radar.py` command


## Lidar Computer

wiselab@lidar-nuc.local

0. Time sync: `sudo ntpdate -u pool.ntp.org`
1. `roslaunch slam ouster.launch`
2. `roslaunch xsens_mti_driver xsens_mti_node.launch`
3. go into `/home/wiselab/catkin_ws/src/data_collection/src/`, then execute `./gt_collect.sh {filename}`

## Cartographer Processing

wiselab@touriga

0. Check that rosbag is good shape: `rosbag info {filename}.bag`
    - If an issue is found: `rosbag reindex {filename}.bag`
1. `roslaunch slam offine_cart_3d.launch bag_filenames:=/home/wiselab/ws/dart/{filename}.bag`
2. `roslaunch slam assets_writer_cart_3d.launch bag_filenames:=/home/wiselab/ws/dart/{filename}.bag pose_graph_filename:=/home/wiselab/ws/dart/{filename}.bag.pbstream`
3. go to `/home/wiselab/catkin_ws/src/slam/src/`, execute `./get_pose.sh {filename}`
