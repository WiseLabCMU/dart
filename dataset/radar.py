"""Radar processing routines."""

from functools import partial
from tqdm import tqdm as default_tqdm

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from jax import numpy as jnp
import jax

from beartype.typing import NamedTuple
from jaxtyping import Complex64, Array, Float64, Float32, Float16


def to_float16(
    arr: Float32[Array, "..."], max_normal: float = 65504.0
) -> tuple[Float16[Array, "..."], float]:
    """Convert array to float16.

    The largest observed value is scaled to the largest normal normal number
    (65504) for 16-bit IEEE floats.
    """
    maxval = np.max(arr)
    return (arr / maxval * max_normal).astype(np.float16)


class AWR1843Boost(NamedTuple):
    """TI AWR1843Boost Radar Configuration.

    Attributes
    ----------
    framelen: Length of each frame; processed as a rolling frame. Corresponds
        to the number of doppler bins available.
    max_range: Maximum range bin to keep (i.e. range decimation).
    chirplen: Number of samples in each chirp.
    chirploops: Number of chirps looped per frame.
    dmax: Doppler bandwidth.
    rmax: Radar max range.
    num_rx: Number of rx antenna.
    num_tx: Number of tx antenna.
    artifact_threshold: Zero-doppler artifact clipping thershold (in percent).
    """

    framelen: int = 256
    max_range: int = 128
    chirplen: int = 512
    chirploops: int = 1
    dmax: float = 1.8949
    rmax: float = 21.5991
    num_rx: int = 4
    num_tx: int = 3
    artifact_threshold: float = 25.0

    @property
    def frame_size(self):
        """Number of complex entries in each frame."""
        return self.chirploops * self.num_rx * self.num_tx * self.chirplen

    @property
    def frame_shape(self):
        """Frame shape."""
        return (self.chirploops * self.num_tx, self.num_rx, self.chirplen)


    def range_doppler_azimuth(
        self, frames: Complex64[Array, "frame antenna range doppler"],
    ) -> Float32[Array, "frame antenna range doppler"]:
        """Process range-doppler-azimuth images.

        Parameters
        ----------
        frames: radar frames to process, with axis order
            [frame #, antenna #, sample #, chirp #]

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
        rda = jnp.fft.fftshift(rda, axes=[1, 3])[..., :self.max_range, :]

        return jnp.abs(rda)

    def remove_artifact(
        self, images: Float32[Array, "frame antenna range doppler"],
    ) -> Float16[Array, "frame antenna range doppler"]:
        """Remove zero-doppler artifact from images.

        Collected range-doppler radar data will have an artifact at close
        ranges in the zero-doppler column which corresponds to the radar return
        of the data collection rig. We subtract the `threshold` percentile
        value from each range bin to get rid of this.
        """
        zero = int(images.shape[3] / 2)
        artifact = jnp.percentile(
            images[..., zero - 1:zero + 2], self.artifact_threshold, axis=0)
        return images.at[..., zero - 1:zero + 2].set(
            images[..., zero - 1:zero + 2]
            - artifact.reshape(1, *artifact.shape))

    def process_data(
        self, chirps: Complex64[np.ndarray, "frame tx rx chirp"],
        timestamps: Float64[np.ndarray, "frame"], stride: int = 64,
        batch_size: int = 64, tqdm=default_tqdm
    ) -> tuple[
        Float16[Array, "idx antenna range doppler"], Float64[Array, "idx"]
    ]:
        """Process radar data.

        Parameters
        ----------
        chirps: Raw chirp data. The number of samples per chirp corresponds to
            the maximum number of range bins available.
        timestamps: timestamps of each frame.
        stride: Stride between successive frames for rolling frame processing.
        batch_size: Frame processing batch. 32/64/128 seems to be the fastest
            based on some rough experiments on a RTX 4090 and 7950X.
        tqdm: Progress bar class.

        Returns
        -------
        [0] processed azimuth-range-doppler images.
        [1] timestamps for each output.
        """
        # Discard the middle TX antenna, and flatten the antennas.
        chirps_flat = chirps[:, (0, 2)].reshape(-1, 8, chirps.shape[3])

        # Sliding window over chirps using stride_tricks; the window contents axis
        # (frame -> doppler) is put in the last axis.
        frames = sliding_window_view(
            chirps_flat, window_shape=self.framelen, axis=0)[::stride]
        image_times = np.mean(
            sliding_window_view(
                timestamps, window_shape=self.framelen)[::stride], axis=1)

        process_func = jax.jit(partial(self.range_doppler_azimuth))

        res = []
        for _ in tqdm(range(int(np.ceil(frames.shape[0] / batch_size)))):
            res.append(process_func(jnp.array(frames[:batch_size])))
            frames = frames[batch_size:]

        res = to_float16(self.remove_artifact(jnp.concatenate(res, axis=0)))

        return res, image_times

    def to_instrinsics(self) -> dict:
        """Export intrinsics configuration file."""
        bin_doppler = self.dmax / self.framelen
        bin_range = self.rmax / self.chirplen
        return {
            "gain": "awr1843boost_az8",
            "n": 512, "d": 256,
            "r": [
                bin_range * 0.5,
                bin_range * (min(self.max_range, self.chirplen) + 0.5),
                min(self.max_range, self.chirplen)
            ],
            "d": [
                -bin_doppler * self.framelen * 0.5,
                bin_doppler * (self.framelen * 0.5 - 1),
                self.framelen
            ]
        }
