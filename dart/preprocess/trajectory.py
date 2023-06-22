"""Pose processing."""

import os
import numpy as np
import pandas as pd

from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import Akima1DInterpolator
from scipy.spatial.transform import Slerp, Rotation

from beartype.typing import NamedTuple
from jaxtyping import Float32, Float64, Bool

from dart import types


class Trajectory(NamedTuple):
    """Sensor trajectory.

    Attributes
    ----------
    spline: Interpolation Akima spline (for 2nd degree continuity).
    slerp: Spherical linear interpolation for rotations.
    """

    spline: Akima1DInterpolator
    slerp: Slerp

    @classmethod
    def from_csv(cls, path: str) -> "Trajectory":
        """Initialize from dataframe output from cartographer.

        Parameters
        ----------
        path: Path to dataset. Should contain a cartographer-formatted output
            csv "trajectory.csv", with columns
            `field.header.stamp` (in ns), `field.transform.translation.{xyz}`,
            `field.transform.rotation.{xyzw}`.
        """
        df = pd.read_csv(os.path.join(path, "trajectory.csv"))

        # Manual 0.5s offset, likely due to buffering the DCA1000
        t_slam = np.array(df["field.header.stamp"]) / 1e9 + 0.5

        xyz = np.array(
            [df["field.transform.translation." + char] for char in "xyz"])
        spline = Akima1DInterpolator(t_slam, xyz.T)

        rot = Rotation.from_quat(np.array(
            [df["field.transform.rotation." + char] for char in "xyzw"]).T)
        slerp = Slerp(t_slam, rot)

        return cls(spline=spline, slerp=slerp)

    def valid_mask(
        self, t: Float64[types.ArrayLike, "N"], window: float = 0.1
    ) -> Bool[types.ArrayLike, "N"]:
        """Get mask of valid timestamps (within the trajectory timestamps)."""
        return (
            (t - window >= self.spline.x[0])
            & (t + window <= self.spline.x[-1]))

    def interpolate(
        self, t: Float64[types.ArrayLike, "N"], window: float = 0.1,
        samples: int = 25
    ) -> dict[str, Float32[types.ArrayLike, "N ..."]]:
        """Calculate poses, averaging along a window.

        A hanning window weighting is used to match the radar FFT processing.

        Parameters
        ----------
        t: Radar frame timestamps, measured at the middle of each frame.
        window: Frame size (end - start), in seconds.
        samples: Number of samples to use for averaging.

        Returns
        -------
        Dictionary with pos, vel, and rot entries.
        """
        window_offsets = np.linspace(-window / 2, window / 2, samples)
        tt = t[..., None] + window_offsets[None, ...]
        hann = np.hanning(samples)

        # Rotation unfortunately does not allow vectorization at this time.
        rot = Rotation.concatenate([
            self.slerp(row).mean(weights=hann) for row in tt])

        return {
            "pos": np.average(
                self.spline(tt), axis=1, weights=hann).astype(np.float32),
            "vel": np.average(
                self.spline.derivative()(tt), axis=1, weights=hann
            ).astype(np.float32),
            "rot": rot.as_matrix().astype(np.float32)
        }

    def postprocess(
        self, velocity: Float32[types.ArrayLike, "N 3"],
        speed_radar: Float32[types.ArrayLike, "N"],
        smoothing: float = -1.0,
        reject_threshold: float = 0.3, reject_kernel: int = 7,
        max_adjustment: float = 0.2, adjust_kernel: int = 15
    ) -> Float32[types.ArrayLike, "N 3"]:
        """Apply smoothing and velocity scaling.

        Parameters
        ----------
        velocity: SLAM system velocity estimates.
        speed: Estimated speed from radar doppler images.
        smoothing: Gaussian filter to apply to the velocity before fusing.
        reject_threshold, reject_kernel: The radar speed estimate is rejected
            when it exceeds the SLAM velocity by `reject_threshold` for more
            than half of a sliding window with size `reject_kernel`.
        max_adjustment: Maximum speed increase permitted by postprocessing.
        adjust_kernel: Speed increase is limited to the median-filtered
            speed estimate with this kernel size.

        Returns
        -------
        Post-processed velocity.
        """
        speed_raw = np.linalg.norm(velocity, axis=1)
        if smoothing > 0.0:
            speed_slam = gaussian_filter1d(speed_raw, sigma=smoothing)
        else:
            speed_slam = speed_raw

        reject = medfilt(
            (speed_slam + reject_threshold < speed_radar).astype(int),
            kernel_size=reject_kernel)
        speed_fused = np.where(
            reject | (speed_slam > speed_radar), speed_slam,
            np.minimum(
                speed_slam + max_adjustment, speed_radar,
                medfilt(speed_radar, kernel_size=adjust_kernel)))

        vel_out = speed_fused[..., None] * velocity / speed_raw[..., None]
        return np.nan_to_num(vel_out, nan=0.0, posinf=0.0, neginf=0.0)
