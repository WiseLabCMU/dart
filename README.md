# DART: Doppler-Aided Radar Tomography

## Setup

1. Install [jax](https://github.com/google/jax). Note that you will need to manually install jax-gpu based on your cuda version, i.e.
    ```sh
    pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    ```
    for CUDA 11.x.

2. Install `libhdf5`:
    ```sh
    sudo apt-get install libhdf5-dev
    ```

3. Install python dependencies:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

TL;DR:
```
python train.py -s data/path/to/sensor.json -o results/output -p data/path/to/data.mat --norm 1e4 --min_speed 0.25 --epochs 20
TARGET=results/output RADIUS=4.0 make eval 
```

- `path/to/sensor.json`: the radar sensor configuration file.
- `path/to/data.mat`: the pose and range-doppler data file.
- `results/output`: output directory.
- `--norm`: normalization factor; divides the input by this number. This will require tuning, and will most likely be somewhere in the range of `1e3` to `1e5`.
- `--min_speed`: speed threshold to reject frames with insufficient velocity (to have enough valid doppler columns). Will require tuning depending on the sampling rate used to generate the frame.

This creates the following files in `results/output`:
```
results/
    output/
        metadata.json   # Model/dataset/training metadata
        model.chkpt     # Model weights checkpoint
        pred.mat        # Predicted range-doppler images for the validation set
        pred.png        # Rendering of 18 random (obs, pred) pairs
        map.png         # Visualization of horizontal slices of the scene 
```

## Available Commands

- `train.py`: train model.
- `evaluate.py`: apply model to all poses in the trace, saving the result to disk. Can't be the same program as `train.py` due to vram limitations - need to completely free training memory allocation before running evaluate.
- `examples.py`: sample and draw 18 random (observed, predicted) pairs of range-doppler images; need to run `evaluate.py` first.
- `map.py`: visualize horizontal slices of the scene for a given area around the origin.
- `video.py`: create video from radar and "camera" frames.

***

## Experiments


### Cabinets

Simulate/Train:
```sh
python simulate.py -s data/cabinets/cabinets.json -o data/cabinets-000/sim.mat -g data/cabinets/map.mat -j data/cabinets-000/cabinets-000.mat
python train.py ngp -s data/cabinets/cabinets.json -o results/cabinets.sim -e 5 --repeat 5 -p data/cabinets-000/sim.mat --min_speed 0.25 --iid
python train.py ngp -s data/cabinets/cabinets.json -o results/cabinets.real -p data/cabinets-000/cabinets-000.mat --norm 1e4 --min_speed 0.25 -e 5 --repeat 5 --iid
python train.py ngpsh -s data/cabinets/cabinets.json -o results/cabinets.ngpsh -p data/cabinets-000/cabinets-000.mat --norm 1e4 --min_speed 0.25 -e 5 --repeat 5 --iid
```

Create video:
```sh
TARGET=results/cabinets.sim make video
TARGET=results/cabinets.real make video
TARGET=results/cabinets.ngpsh make video
python video.py -p results/cabinets.sim results/cabinets.real -f 30 -s 512 -o results/cabinets.mp4
```

### Motion Stage

Simulate:
```sh
python simulate.py -s data/linear1/linear1.json -o data/linear1/sim.mat -g data/linear1/map.mat -j data/linear1/linear1.mat
```

Train:
```sh
python train.py ngp -s data/linear1/linear1-fixed.json -o results/linear1.sim -p data/linear1/sim.mat --min_speed 0.005 -b 512 -e 5 --repeat 10 --iid
python train.py ngp -s data/linear1/linear1-fixed.json -o results/linear1.real -p data/linear1/linear1.mat --norm 1e6 --min_speed 0.005 -b 512 -e 5 --repeat 10 --iid
```

Create video:
```sh
TARGET=results/linear1.sim make video
TARGET=results/linear1.real make video
python video.py -p results/linear1.sim results/linear1.real -f 15 -s 512 -o results/linear1.mp4
```

### Coloradar

```sh
python train.py ngp -s data/coloradar/coloradar-short.json -o results/coloradar0-short -p data/coloradar/coloradar0.mat --norm 1e3 --min_speed 1.0 --base 2.0 --iid --repeat 5 -e 10
TARGET=results/coloradar0-short make eval
```

```sh
python train.py ngp -s data/coloradar/coloradar-short.json -o results/coloradar1 -p data/coloradar/coloradar1.mat --norm 1e4 --min_speed 1.0 --base 2.0 --iid --repeat 5 -e 10
TARGET=results/coloradar1 make eval
```