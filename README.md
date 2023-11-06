# DART: Doppler-Aided Radar Tomography

Implementation of *(Solving) Two Birds with one DART: Implicit Doppler Tomography for Radar Mapping and Novel View Synthesis*

![DART method overview.](docs/dart.svg)

## Setup

0. Ensure that you have python (`>=3.8`), CUDA (`>=11.8`), and CUDNN.

1. Install [jax](https://github.com/google/jax). Note that you will need to manually install jax-gpu to match the cuda version:
    ```sh
    pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    ```
    for CUDA 11.x.

    **NOTE**: jax is not included in `requirements.txt` due to requiring CUDA-dependent manual installation.

2. Install `libhdf5`:
    ```sh
    sudo apt-get -y install libhdf5-dev
    ```

3. Install python dependencies:
    ```sh
    pip install -r requirements.txt
    ```
    - Use Python 3.11, CUDA 11.8, Jax 0.4.10, and ```pip install -r requirements-pinned.txt``` to get the exact version of dependencies that we used.

## Usage

TL;DR:
```sh
TARGET=output DATASET=cichallway make experiment
```

With arguments:
```sh
TARGET=output METHOD=ngp DATASET=cichallway FLAGS="--min_speed 0.25 --epochs 5" make experiment
```

- `METHOD=ngp`: model to train (`ngp`, `ngpsh`, `grid`).
- `DATASET=path/to/dataset`: dataset to use, organized as follows:
    ```
    data/path/to/dataset/
        sensor.json       # Sensor configuration file
        data.h5          # Data file with pose and range-doppler images.
    ```
    *Note*: `sensor.json` and `dataset.h5` can also be specified by `-s path/to/dataset.json` and `-p path/to/dataset.h5`.
- `TARGET=path/to/output`: save results (checkpoints, evaluation, and configuration) to `results/path/to/output`.
- `FLAGS=...`: arguments to pass to `train.py`; see `python train.py -h` and `python train.py ngp -h`, replacing `ngp` with the target method.

This creates the following files in `results/output`:
```sh
results/
    output/
        metadata.json       # Model/dataset/training metadata
        model.chkpt         # Model weights checkpoint
        pred.h5             # Predicted range-doppler images
        cam.h5              # Virtual camera renderings for the trajectory
        map.h5              # Map of the scene sampled at 25 units/meter
        output.video.mp4    # Output camera + radar video
        output.map.mp4      # Video where each frame is a horizontal slice
```

Multiple models on the same trajectory can also be combined into a single output video:
```sh
python manage.py video -p results/output results/output2 ... -f 30 -s 512 -o results/video.mp4
```

## Available Commands

- `train.py`: train model.
- `manage.py`: evaluation and visualization tools:
    - `evaluate`: apply model to all poses in the trace, saving the result to disk. Can't be the same program as `train.py` due to vram limitations - need to completely free training memory allocation before running evaluate.
    - `examples`: sample and draw 18 random (observed, predicted) pairs of range-doppler images; need to run `evaluate` first.
    - `map`: visualize horizontal slices of the scene for a given area around the origin.
    - `simulate`: create simulated dataset.
    - `video`: create video from radar and "camera" frames; need to run `evaluate -a` and `evaluate -ac` first.
