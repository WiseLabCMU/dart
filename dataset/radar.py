"""Radar processing functions."""

from functools import partial
from tqdm import tqdm as default_tqdm

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from jax import numpy as jnp
import jax

from jaxtyping import Complex64, Array, Float64, Float32, Float16


def range_doppler_azimuth(
    frames: Complex64[Array, "frame antenna range doppler"],
    max_range: int = 128
) -> Float32[Array, "frame antenna range doppler"]:
    """Process range-doppler-azimuth images.

    Parameters
    ----------
    frames: radar frames to process, with axis order
        [frame #, antenna #, sample #, chirp #]
    max_range: number of range bins to keep.

    Returns
    -------
    Range-doppler-azimuth array (magnitude only).
    """
    # Hanning window to suppress range, doppler bleed
    window = (
        jnp.hanning(frames.shape[3]).reshape(1, -1)
        * jnp.hanning(frames.shape[2]).reshape(-1, 1))

    # Range-doppler: fft along frame (-> doppler) and chirp (-> range)
    rd = jnp.fft.fft2(frames * window.reshape(1, 1, *window.shape))
    # Azimuth: fft along antenna
    rda = jnp.fft.fft(rd, axis=1)

    # fftshift along antenna, doppler axes
    rda = jnp.fft.fftshift(rda, axes=[1, 3])[..., :max_range, :]

    return jnp.abs(rda)


def remove_artifact(
    images: Float32[Array, "frame antenna range doppler"]
) -> Float16[Array, "frame antenna range doppler"]:
    """Remove zero-doppler artifact from images.

    Collected range-doppler radar data will have an artifact at close ranges
    in the zero-doppler column which corresponds to the radar return of the
    data collection rig. We subtract the minimum observed value from each range
    to get rid of this.
    """
    zero = int(images.shape[3] / 2)
    artifact = jnp.min(images[..., zero - 1:zero + 2], axis=0)
    return images.at[..., zero - 1:zero + 2].set(
        images[..., zero - 1:zero + 2] - artifact.reshape(1, *artifact.shape))


def to_float16(
    arr: Float32[Array, "..."], max_normal: float = 65504.0
) -> tuple[Float16[Array, "..."], float]:
    """Convert array to float16.

    The largest observed value is scaled to the largest normal normal number
    (65504) for 16-bit IEEE floats. The effective scale factor applied
    is also returned.
    """
    maxval = np.max(arr)
    return (arr / maxval * max_normal).astype(np.float16), maxval / max_normal


def calculate_intrinsics(
    framelen: int = 256, max_range: int = 128, chirplen: int = 512,
    dmax: float = 1.8949, rmax: float = 21.5991
) -> dict:
    """Get radar intrinsics."""
    bin_doppler = dmax / framelen
    bin_range = rmax / chirplen
    return {
        "gain": "awr1843boost_az8",
        "n": 512, "d": 256,
        "r": [
            bin_range * 0.5,
            bin_range * (min(max_range, chirplen) + 0.5),
            min(max_range, chirplen)
        ],
        "d": [
            -bin_doppler * framelen * 0.5,
            bin_doppler * (framelen * 0.5 - 1),
            framelen
        ]
    }


def process_data(
    chirps: Complex64[np.ndarray, "frame tx rx chirp"],
    timestamps: Float64[np.ndarray, "frame"],
    max_range: int = 128, framelen: int = 256, stride: int = 64,
    batch_size: int = 64, tqdm=default_tqdm
) -> tuple[
    Float16[Array, "idx antenna range doppler"],
    Float64[Array, "idx"], dict
]:
    """Process radar data.

    Parameters
    ----------
    chirps: Raw chirp data. The number of samples per chirp corresponds to
        the maximum number of range bins available.
    timestamps: timestamps of each frame.
    max_range: Maximum range bin to keep (i.e. range decimation).
    framelen: Length of each frame; processed as a rolling frame. Corresponds
        to the number of doppler bins available.
    stride: Stride between successive frames for rolling frame processing.
    batch_size: Frame processing batch. 32/64/128 seems to be the fastest based
        on some rough experiments on a RTX 4090 and 7950X.
    tqdm: Progress bar class.

    Returns
    -------
    [0] processed azimuth-range-doppler images.
    [1] timestamps for each output.
    [2] radar intrinsics for this dataset.
    """
    # Discard the middle TX antenna, and flatten the antennas.
    chirps_flat = chirps[:, (0, 2)].reshape(-1, 8, chirps.shape[3])

    # Sliding window over chirps using stride_tricks; the window contents axis
    # (frame -> doppler) is put in the last axis.
    frames = sliding_window_view(
        chirps_flat, window_shape=framelen, axis=0)[::stride]
    image_times = np.mean(
        sliding_window_view(timestamps, window_shape=framelen)[::stride],
        axis=1)

    process_func = jax.jit(partial(range_doppler_azimuth, max_range=max_range))

    res = []
    for _ in tqdm(range(int(np.ceil(frames.shape[0] / batch_size)))):
        res.append(process_func(jnp.array(frames[:batch_size])))
        frames = frames[batch_size:]

    res, scale = to_float16(remove_artifact(jnp.concatenate(res, axis=0)))
    cfg = calculate_intrinsics(
        framelen=framelen, max_range=max_range, chirplen=chirps.shape[3])
    cfg["scale"] = float(scale)

    return res, image_times, cfg
