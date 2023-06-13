"""Result convenience wrapper."""

import os
import json
import h5py
from scipy.io import savemat

from beartype.typing import Optional, Any
from jaxtyping import Integer, Array

from .dart import DART
from .dataset import load_arrays, trajectory
from . import types


class DartResult:
    """DART experiment results."""

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

    def map(self) -> Any:
        """Get map file."""
        return load_arrays(os.path.join(self.path, "map.mat"))

    def save(self, path: str, contents: dict) -> None:
        """Save contents as a hdf5 file in this result scope."""
        with h5py.File(os.path.join(self.path, path), 'w') as hf:
            for k, v in contents.items():
                hf.create_dataset(k, data=v)
            hf.close()

    def load(self, path: str, keys: Optional[list[str]] = None) -> Any:
        """Load file inside this result."""
        return load_arrays(os.path.join(self.path, path), keys=keys)
