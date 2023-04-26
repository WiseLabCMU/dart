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
        pred.mat        # Predicted range-doppler images
        pred.png        # Rendering of 18 random (obs, pred) pairs
        map.png         # Visualization of horizontal slices of the scene 
```

## Available Commands

- `train.py`: train model.
- `evaluate.py`: apply model to all poses in the trace, saving the result to disk. Can't be the same program as `train.py` due to vram limitations - need to completely free training memory allocation before running evaluate.
- `examples.py`: sample and draw 18 random (observed, predicted) pairs of range-doppler images; need to run `evaluate.py` first.
- `map.py`: visualize horizontal slices of the scene for a given area around the origin.

***

Cabinets (sim):
```sh
python train.py ngp -s data/cabinets/cabinets.json -o results/cabinets-sim -e 2 -p data/cabinets/sim.mat --weight sqrt --min_speed 0.25
python evaluate.py -p results/cabinets-sim; python examples.py results/cabinets-sim; python map.py -p results/cabinets-sim
```

Cabinets (real):
```sh
python train.py ngp -s data/cabinets/cabinets-fixed.json -o results/cabinets.000 -p data/cabinets-sep/cabinets-000/cabinets-000.mat --norm 1e4 --min_speed 0.25 -e 25; TARGET=results/cabinets.000 RADIUS=4 make eval```
```

Motion Stage:
```sh
python train.py ngp -s data/linear1/linear1-fixed.json -o results/linear1 -p data/linear1/linear1.mat --norm 1e6 --min_speed 0.005 -b 512 -e 10 --repeat 5; TARGET=results/linear1 RADIUS=0.6 make eval
```
