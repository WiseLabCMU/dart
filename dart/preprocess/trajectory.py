"""Pose processing."""

import os
import numpy as np
import pandas as pd

from scipy.interpolate import Akima1DInterpolator
from scipy.spatial.transform import Slerp, Rotation

from beartype.typing import NamedTuple
from jaxtyping import Float32, Float64, Bool


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

        t_slam = np.array(df["field.header.stamp"]) / 1e9

        # tmp
        t_slam = t_slam + 0.5

        xyz = np.array(
            [df["field.transform.translation." + char] for char in "xyz"])
        spline = Akima1DInterpolator(t_slam, xyz.T)

        rot = Rotation.from_quat(np.array(
            [df["field.transform.rotation." + char] for char in "xyzw"]).T)
        slerp = Slerp(t_slam, rot)

        return cls(spline=spline, slerp=slerp)

    def valid_mask(self, t: Float64[np.ndarray, "N"]) -> Bool[np.ndarray, "N"]:
        """Get mask of valid timestamps (within the trajectory timestamps)."""
        return (t >= self.spline.x[0]) & (t <= self.spline.x[-1])

    def interpolate(
        self, t: Float64[np.ndarray, "N"], window: float = 0.1,
        samples: int = 25
    ) -> dict[str, Float32[np.ndarray, "N ..."]]:
        """Calculate poses, averaging along a window.

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
        samples = t[..., None] + window_offsets[None, ...]

        # Rotation unfortunately does not allow vectorization at this time.
        rot = Rotation.concatenate([self.slerp(row).mean() for row in samples])

        return {
            "pos": np.mean(self.spline(samples), axis=1),
            "vel": np.mean(self.spline.derivative()(samples), axis=1),
            "rot": rot.as_matrix()
        }
