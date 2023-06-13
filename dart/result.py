"""Result convenience wrapper."""

import os
import json
import numpy as np
import h5py
import matplotlib as mpl

from beartype.typing import Optional, Any, Union
from jaxtyping import Integer, Array, Float

from .dart import DART
from .dataset import load_arrays, trajectory
from . import types


class DartResult:
    """DART experiment results.

    Parameters
    ----------
    path: result path.
    """

    METADATA = "metadata.json"
    MAP = "map.h5"
    WEIGHTS = "model.npz"
    CAMERA = "cam.h5"
    RADAR = "rad.h5"

    def __init__(self, path: str) -> None:
        self.path = path

        _meta = os.path.join(self.path, self.METADATA)
        if not os.path.exists(_meta):
            raise FileNotFoundError(
                "Result path does exist (could not find {}).".format(_meta))

        with open(_meta) as f:
            self.metadata = json.load(f)

    def dart(self) -> DART:
        """Get DART object."""
        return DART.from_config(**self.metadata)

    def trajectory_dataset(
        self, subset: Optional[Integer[Array, "nval"]] = None
    ) -> types.Dataset:
        """Load trajectory dataset."""
        return trajectory(self.metadata["dataset"]["path"], subset=subset)

    def data(self, keys: Optional[list[str]] = None) -> Any:
        """Get dataset file."""
        return load_arrays(self.metadata["dataset"]["path"], keys=keys)

    def save(self, path: str, contents: dict) -> None:
        """Save contents as a hdf5 file in this result scope."""
        with h5py.File(os.path.join(self.path, path), 'w') as hf:
            for k, v in contents.items():
                hf.create_dataset(k, data=v)
            hf.close()

    def load(self, path: str, keys: Optional[list[str]] = None) -> Any:
        """Load file inside this result."""
        return load_arrays(os.path.join(self.path, path), keys=keys)

    @staticmethod
    def colorize_map(
        arr: Float[Array, "..."], sigma: bool = True,
        clip: tuple[float, float] = (1.0, 99.0), conv: int = 0
    ) -> Float[Array, "... 3"]:
        """Colorize a sigma or alpha map.

        Sigma maps are colorized by clipping to the provided percentiles, then
        scaling the upper percentile to 1.0 (keeping 0.0).

        Alpha maps are colorized directly on a 0-1 scale.

        Parameters
        ----------
        arr: input array with 2 or 3 dimensions.
        sigma: whether this is a sigma map or a log-alpha map.
        clip: percentiles to clip (sigma only).
        conv: convolutional smoothing in the z-axis, if applicable.

        Returns
        -------
        8-bit RGB-encoded color map.
        """
        if len(arr.shape) == 3 and conv > 0:
            arr = sum(  # type: ignore
                arr[..., i:arr.shape[2] - (conv - i)] for i in range(conv + 1)
            ) / (conv + 1)

        if sigma:
            lower, upper = np.percentile(arr, clip)
            arr = np.clip(arr, max(lower, 0.0), upper) / upper
        else:
            arr = np.exp(arr)  # type: ignore

        return (mpl.colormaps['viridis'](arr)[..., :3] * 255).astype(np.uint8)
