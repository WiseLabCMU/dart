"""Radar processing routines."""

import json
import math
from functools import partial

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

    The array is clipped to the largest normal number (`max_normal=65504`)
    for 16-bit IEEE floats before casting.
    """
    return jnp.minimum(arr, max_normal).astype(jnp.float16)


class AWR1843Boost(NamedTuple):
    """TI AWR1843Boost Radar Configuration.

    Attributes
    ----------
    framelen: Number of chirps in each frame; processed as a rolling buffer.
        Corresponds to the number of doppler bins available.
    stride: Frame chirp stride (advance by `stride` chirps for each new frame.)
    max_range: Maximum range bin to keep (i.e. range decimation).
    chirplen: Number of samples in each chirp.
    chirploops: Number loops per chirp.
    dmax: Doppler bandwidth.
    rmax: Radar max range.
    num_rx: Number of rx antenna.
    num_tx: Number of tx antenna.
    scan_dt: Period of radar chirps, in seconds.
    artifact_threshold: Zero-doppler artifact clipping threshold (in percent).
    scale: Radar magnitude scale factor to divide by before casting to float16;
        the resulting value is clipped to the largest normal number (65504).
    calibration: Signed difference between range bins and actual range;
        positive `calibration` corresponds to objects appearing further
        than they are physically.
    """

    framelen: int = 256
    stride: int = 64
    max_range: int = 128
    chirplen: int = 512
    chirploops: int = 1
    dmax: float = 1.8949
    rmax: float = 21.5991
    num_rx: int = 4
    num_tx: int = 2
    scan_dt: float = 0.001
    artifact_threshold: float = 50.0
    scale: float = 1e6
    calibration: float = -0.05671

    @classmethod
    def from_json(cls, path: str) -> "AWR1843Boost":
        """Create from JSON configuration file."""
        with open(path) as f:
            return cls(**json.load(f))

    @property
    def chirp_size(self):
        """Number of complex entries in each chirp."""
        return self.chirploops * self.num_rx * self.num_tx * self.chirplen

    @property
    def chirp_shape(self):
        """Radar chirp shape."""
        return (self.chirploops * self.num_tx, self.num_rx, self.chirplen)

    @property
    def frame_time(self):
        """Time per frame (in seconds)."""
        return self.framelen * self.scan_dt

    @property
    def image_shape(self):
        """Radar range-azimuth-doppler image shape."""
        return (self.max_range, self.framelen, 8)

    @property
    def min_range(self):
        """Minimum valid range bin, after accounting for calibration."""
        range_bin = self.rmax / self.chirplen
        return max(0, math.ceil(-self.calibration / range_bin))

    def estimate_speed(
        self, rda: Float32[Array, "frame range doppler antenna"],
        percentile: float = 99.75, min_count: int = 10
    ) -> Float32[Array, "frame"]:
        """Estimate speed for this frame using doppler spectrum.

        Parameters
        ----------
        rda: range-doppler-azimuth image.
        percentile: clipping percentile for "present" objects.
        min_count: minimum number of thresholded bins across range/azimuth to
            be counted as a present doppler column.

        Returns
        -------
        Estimated speed.
        """
        nd_half = int(self.framelen / 2)

        # Threshold for each frame, antenna (over=1,2)
        threshold = jnp.percentile(
            rda, percentile, axis=(1, 2))[:, None, None, ...]
        # Count across range, antenna (over=2,3)
        valid = jnp.sum(rda > threshold, axis=(1, 3)) > min_count

        left = jnp.argmax(valid[:, :nd_half], axis=1)
        right = jnp.argmax(valid[:, nd_half:][:, ::-1], axis=1)
        speed_nd = jnp.minimum(left, right)

        return (nd_half - speed_nd) / nd_half * self.dmax / 2

    def range_doppler_azimuth(
        self, frames: Complex64[Array, "frame antenna range doppler"],
    ) -> tuple[
        Float32[Array, "frame range_clipped doppler antenna"],
        Float32[Array, "frame"]
    ]:
        """Process range-doppler-azimuth images.

        A hanning window is used along range and doppler axes to suppress
        range and doppler bleed.

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

        speed = self.estimate_speed(rda)
        range_clipped = rda[:, self.min_range:self.max_range + self.min_range]

        return range_clipped, speed

    def remove_artifact(
        self, images: Float32[Array, "frame range doppler antenna"],
    ) -> Float32[Array, "frame range doppler antenna"]:
        """Remove zero-doppler artifact from images.

        Collected range-doppler radar data will have an artifact at close
        ranges in the zero-doppler column which corresponds to the radar return
        of the data collection rig. We subtract the `threshold` percentile
        value from each range bin to get rid of this.
        """
        zero = int(images.shape[2] / 2)
        artifact = jnp.percentile(
            images[..., zero - 1:zero + 2, :], self.artifact_threshold, axis=0)
        removed = jnp.maximum(
            images[..., zero - 1:zero + 2, :]
            - artifact.reshape(1, *artifact.shape), 0)
        return images.at[..., zero - 1:zero + 2, :].set(removed)

    def process_timestamps(
        self, timestamps: Float64[types.ArrayLike, "num_chirps"]
    ) -> Float64[types.ArrayLike, "num_frames"]:
        """Compute timestamps for each image when frames are processed.

        NOTE: Timestamps are measured at the mean chirp corresponding to each
        frame. The mean chirp is used to smooth the effect of 15ms jitter which
        we observe in the timestamp, likely due to software timestamping
        interacting with scheduling effects in windows on the data capture
        computer.

        Parameters
        ----------
        timestamps: timestamps of each chirp, measured at the beginning.

        Returns
        -------
        Timestamp of each radar frame.
        """
        return np.mean(sliding_window_view(
            timestamps, window_shape=self.framelen)[::self.stride], axis=1)

    def process_data(
        self, chirps: Complex64[types.ArrayLike, "frame tx rx chirp"],
        batch_size: int = 64
    ) -> tuple[
        Float16[types.ArrayLike, "idx antenna range doppler"],
        Float32[types.ArrayLike, "idx"]
    ]:
        """Process radar data.

        Parameters
        ----------
        chirps: Raw chirp data. The number of samples per chirp corresponds to
            the maximum number of range bins available.
        batch_size: Frame processing batch. 32/64/128 seems to be the fastest
            based on some rough experiments on a RTX 4090 and 7950X.

        Returns
        -------
        [0] processed azimuth-range-doppler images.
        [1] estimated speed of this frame based on max observed doppler.
        """
        # Discard the middle TX antenna, and flatten the antennas.
        chirps_flat = chirps.reshape(-1, 8, chirps.shape[3])

        # Sliding window over chirps using stride_tricks; the window contents
        # axis (frame -> doppler) is put in the last axis.
        frames = sliding_window_view(
            chirps_flat, window_shape=self.framelen, axis=0)[::self.stride]

        process_func = jax.jit(partial(self.range_doppler_azimuth))
        res, speed = [], []
        for _ in range(int(np.ceil(frames.shape[0] / batch_size))):
            r, s = process_func(jnp.array(frames[:batch_size]))
            res.append(r)
            speed.append(s)
            frames = frames[batch_size:]
        res_arr = self.remove_artifact(jnp.concatenate(res, axis=0))
        return to_float16(res_arr / self.scale), jnp.concatenate(speed)

    def to_instrinsics(self) -> dict:
        """Export intrinsics configuration file."""
        bin_doppler = self.dmax / self.framelen
        bin_range = self.rmax / self.chirplen
        return {
            "gain": "awr1843boost_az8",
            "r": [
                bin_range * self.min_range + self.calibration,
                bin_range * min(self.min_range + self.max_range, self.chirplen)
                + self.calibration,
                min(self.max_range, self.chirplen)
            ],
            "d": [
                -bin_doppler * self.framelen * 0.5,
                bin_doppler * (self.framelen * 0.5 - 1),
                self.framelen
            ]
        }
