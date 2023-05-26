# DART: Doppler-Aided Radar Tomography

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
    sudo apt-get install libhdf5-dev
    ```

3. Install python dependencies:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

TL;DR:
```sh
python train.py ngp -p path/to/dataset -o results/output --norm 1e4 --min_speed 0.25 --epochs 5
TARGET=results/output make video
```

For example:
```
python train.py ngp -p data/aframe -o results/aframe --norm 1e4 --min_speed 0.25 --epochs 5
TARGET=results/aframe make video
```

- `ngp`: model to train (`ngp`, `ngpsh`, `grid`).
- `path/to/dataset`: path to dataset files, organized as follows:
    ```
    path/to/dataset/
        dataset.json    # Sensor configuration file
        dataset.mat     # Data file with pose and range-doppler images.
    ```
    *Note*: `dataset.json` and `dataset.mat` can also be specified by `-s path/to/dataset.json` and `-p path/to/dataset.mat`.
- `results/output`: output directory.
- `--norm`: normalization factor; divides the input by this number. This will require tuning, and will most likely be somewhere in the range of `1e3` to `1e5`.
- `--min_speed`: speed threshold to reject frames with insufficient velocity (to have enough valid doppler columns). Will require tuning depending on the sampling rate used to generate the frame.

This creates the following files in `results/output`:
```
results/
    output/
        metadata.json   # Model/dataset/training metadata
        model.chkpt     # Model weights checkpoint
        pred_all.mat    # Predicted range-doppler images
        cam_all.mat     # Virtual camera renderings for the trajectory
        pred.png        # Rendering of 18 random (obs, pred) pairs
        map.png         # Visualization of horizontal slices of the scene
    output.mp4
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
