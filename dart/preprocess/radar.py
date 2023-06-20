"""Radar processing routines."""

from functools import partial
from tqdm import tqdm as default_tqdm

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from jax import numpy as jnp
import jax

from beartype.typing import NamedTuple
from jaxtyping import Complex64, Array, Float64, Float32, Float16
from dart import types


def to_float16(
    arr: Float32[Array, "..."], max_normal: float = 65504.0
) -> Float16[Array, "..."]:
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
    artifact_threshold: Zero-doppler artifact clipping threshold (in percent).
    """

    framelen: int = 512
    max_range: int = 128
    chirplen: int = 512
    chirploops: int = 1
    dmax: float = 1.8949 * 2
    rmax: float = 21.5991
    num_rx: int = 4
    num_tx: int = 2
    artifact_threshold: float = 50.0

    @property
    def frame_size(self):
        """Number of complex entries in each frame."""
        return self.chirploops * self.num_rx * self.num_tx * self.chirplen

    @property
    def frame_shape(self):
        """Radar chirp frame shape."""
        return (self.chirploops * self.num_tx, self.num_rx, self.chirplen)

    @property
    def image_shape(self):
        """Radar range-azimuth-doppler image shape."""
        return (self.max_range, self.framelen, 8)

    def estimate_speed(
        self, rda: Float32[Array, "frame range doppler antenna"]
    ) -> Float32[Array, "frame"]:
        """Estimate speed for this frame using doppler spectrum."""
        nd_half = int(self.framelen / 2)

        # Spectrum across doppler bins; folded in half
        _spectrum = np.sum(rda, axis=(1, 3))
        spectrum = _spectrum[:, :nd_half] + _spectrum[:, nd_half:][:, ::-1]

        threshold = jnp.min(spectrum, axis=1).reshape(-1, 1)
        speed_nd = jnp.argmax(
            (jnp.diff(spectrum, axis=1) > threshold)
            & (spectrum[:, :-1] > (threshold * 2)), axis=1)

        return (nd_half - speed_nd) / nd_half * self.dmax / 2

    def range_doppler_azimuth(
        self, frames: Complex64[Array, "frame antenna range doppler"],
    ) -> tuple[
        Float32[Array, "frame range_clipped doppler antenna"],
        Float32[Array, "frame"]
    ]:
        """Process range-doppler-azimuth images.

        Parameters
        ----------
        frames: radar frames to process, with axis order
            [frame #, antenna #, sample #, chirp #]

        Returns
        -------
        [0] Range-doppler-azimuth array (magnitude only).
        [1] Estimated speed of this frame.
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
        rda = jnp.fft.fftshift(rda, axes=[1, 3])
        # Discard complex, change to frame-range-doppler-azimuth order
        rda = jnp.moveaxis(jnp.abs(rda), [1, 2, 3], [3, 1, 2])

        return rda[:, :self.max_range], self.estimate_speed(rda)

    def remove_artifact(
        self, images: Float32[types.ArrayLike, "frame range doppler antenna"],
    ) -> Float32[types.ArrayLike, "frame range doppler antenna"]:
        """Remove zero-doppler artifact from images.

        Collected range-doppler radar data will have an artifact at close
        ranges in the zero-doppler column which corresponds to the radar return
        of the data collection rig. We subtract the `threshold` percentile
        value from each range bin to get rid of this.
        """
        zero = int(images.shape[2] / 2)
        artifact = jnp.percentile(
            images[..., zero - 1:zero + 2, :], self.artifact_threshold, axis=0)
        removed = np.maximum(
            images[..., zero - 1:zero + 2, :]
            - artifact.reshape(1, *artifact.shape), 0)
        return images.at[..., zero - 1:zero + 2, :].set(removed)

    def process_data(
        self, chirps: Complex64[types.ArrayLike, "frame tx rx chirp"],
        timestamps: Float64[types.ArrayLike, "frame"], stride: int = 128,
        batch_size: int = 64
    ) -> tuple[
        Float16[types.ArrayLike, "idx antenna range doppler"],
        Float64[types.ArrayLike, "idx"],
        Float32[types.ArrayLike, "idx"]
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
        [2] estimated speed of this frame.
        """
        # Discard the middle TX antenna, and flatten the antennas.
        chirps_flat = chirps.reshape(-1, 8, chirps.shape[3])

        # Sliding window over chirps using stride_tricks; the window contents
        # axis (frame -> doppler) is put in the last axis.
        frames = sliding_window_view(
            chirps_flat, window_shape=self.framelen, axis=0)[::stride]
        image_times = np.median(
            sliding_window_view(
                timestamps, window_shape=self.framelen)[::stride], axis=1)

        process_func = jax.jit(partial(self.range_doppler_azimuth))
        res, speed = [], []
        for _ in range(int(np.ceil(frames.shape[0] / batch_size))):
            r, s = process_func(jnp.array(frames[:batch_size]))
            res.append(r)
            speed.append(s)
            frames = frames[batch_size:]
        res_arr = self.remove_artifact(jnp.concatenate(res, axis=0))
        return to_float16(res_arr), image_times, jnp.concatenate(speed)

    def to_instrinsics(self) -> dict:
        """Export intrinsics configuration file."""
        bin_doppler = self.dmax / self.framelen
        bin_range = self.rmax / self.chirplen
        return {
            "gain": "awr1843boost_az8",
            "n": 512, "k": 256,
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
