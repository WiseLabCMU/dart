# DART: Doppler-Aided Radar Tomography

Implementation of *DART: Implicit Doppler Tomography for Radar Mapping and Novel View Synthesis*

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
    ```sh
    data/path/to/dataset/
        sensor.json             # Sensor configuration file
        data.h5                 # Data file with pose and range-doppler images.
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

## Dataset Format

Our DART implementation loads `.h5` files. DART expects the following arrays, where axis 0 in each array represents a data point, which consists of a doppler column:

| Key, Type | Description |
| --- | ----------- |
| `x : Float[n, 3]` | Radar position (4); axis 0/1 are horizontal, and axis 2 is vertical. |
| `A : Float[n, 3, 3]` | Radar rotation (1) matrix (radar orientation -> front-left-up) |
| `v : Float[n, 3]` | Radar velocity direction vector (2) |
| `s : Float[n]` | Radar speed (2,4) |
| `p, q: Float[n, 3]` | Vectors which form an orthonormal basis along with `v` |
| `weight: Float[n]` | Integration weight of each sample in this doppler column (3) |
| `doppler: Float[n]` | Doppler velocity (4) of this column |
| `doppler_idx: Int[n]` | Local index of this particular doppler velocity (metadata only) |
| `frame_idx: Int[n]` | Global index of this particular frame (metadata only) |
| `i: Int[n]` | Global index of this doppler column (metadata only) |
| `rad: Float[n, range, azimuth]` | Actual values of this doppler column (4) |

Notes:
1. Our coordinate convention is FLU (Front-Left-Up), where the sensor FOV is centered around +x, +y is to the left of +x, and +z is straight up.
2. `v` is a normal vector, describes the velocity along with `s`, and forms an orthonormal basis along with `p, q`.
3. The integration weight captures (and pre-computes) the impact of arc length and speed on numerical integration.
4. Units are in m, m/s where applicable. The actual radar bin value does not have a specific unit due to the difficulty of propagating interpretable scaling magnitudes through the entire radar collection and processing pipeline.

These values also include some redundant precomputed information. If working with the output from our data collection platform, [rover](https://github.com/thetianshuhuang/rover) (`radar.h5` and `trajectory.h5`), `data.h5` can be created using `dataset.py`:
```sh
python manage.py dataset -p <path_to_dataset> --val <0.2>
```

## Available Commands

See `-h` for each command/subcommand for more details.

`train.py`: train model; each subcommand is a different model class.
- `grid`: plenoxels-style grid.
- `ngp`: non-view-dependent NGP-style neural hash grid.
- `ngpsh`: NGP-style neural hash with plenoctrees-style view dependence (what DART uses).
- `ngpsh2`: NGP-style neural hash with nerfacto-style view dependence.

`manage.py`: evaluation and visualization tools:
- `simulate`: create simulated dataset from a ground truth reflectance grid (e.g. the `map.npz` file obtained from LIDAR.)
- `evaluate`: apply model to all poses in the trace, saving the result to disk. Can't be the same program as `train.py` due to vram limitations - need to completely free training memory allocation before running evaluate.
- `video`: create video from radar and "camera" frames; need to run `evaluate -a` and `evaluate -ac` first.
- `map`: evaluate DART model in a grid.
- `slice`: render MRI-style tomographic slices, and write each slice to a frame in a video; need to run `map` first.
- `metrics`: compute validation-set SSIM for range-doppler-azimuth images.
- `compare`: create a side-by-side video comparison of different methods (and/or ground truth).
- `dataset`: create a filtered train/val.
- `psnr`: calculate reference PSNR for gaussian noise.
