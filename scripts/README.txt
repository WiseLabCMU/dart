power supply charges via USB-C
power supply power button
hold power supply AC button until AC is shown on screen
connect laptop to dart-5 (wiselab2023)
AFTER laptop is connected to wifi, plug in NUC + radar and press power on NUC
run RDP on laptop
connect to win-qbb0j8pltgl.local
username Administrator
password mmwave76GHz
run cmd
cd git/dart/scripts
conda activate dart-datacollect
python collectall.py -o C:/Users/Administrator/Desktop/dartdata/<DATASET_NAME>
wait about 60 seconds until mmwave GUI closes
should see "flushing 200 poses" AND "flushing 8192 packets"
if radar packets are not showing up (this happens on first boot of NUC):
- ctrl-C
- open task manager
- find DCA1000EVM exe
- end task
- start again from the python collectall command
walk slowly with battery in backpack
stay within wifi range of laptop
scan with varying heights and facing directions
try to move smoothly
when done (a few minutes, up to 20)
ctrl-c
unplug radar power
wait 60 seconds
plug in external drive to NUC
copy C:/Users/Administrator/Desktop/dartdata/<DATASET_NAME> to HEADCOUNT/dartdata/<DATASET_NAME>
safely remove drive
unplug drive
collect another dataset if you want (remember to plug radar back in first)
when done, shut down NUC and power off power supply
plug in external drive to PC
cd dart/scripts
conda activate dart-datacollect
python radar/preprocess_radarpackets.py -d /media/john/HEADCOUNT/dartdata/<DATASET_NAME>
ignore output, wait until finished
open Matlab
change directory to dart/matlab directory
open dart_preprocess.m
change dataset name/directory
choose range/doppler decimation appropriately, etc
for T265 traces, ensure USE_T265 is true
click "run" and wait for it to finish
copy <DATASET_NAME>.mat and <DATASET_NAME>.json to dart/data/<DATASET_NAME>
optional: load dartdata/<DATASET_NAME>/dbg.mat and run compare_simulation.mat
