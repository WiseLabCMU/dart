"""Result convenience wrapper."""

import os
import json
from scipy.io import savemat

from beartype.typing import Optional, Any
from jaxtyping import Integer, Array

from .dart import DART
from .dataset import load_arrays, trajectory
from . import types


class DartResult:
    """DART experiment results."""

    def __init__(self, path: str) -> None:
        self.path = path

        _meta = os.path.join(self.path, "metadata.json")
        if not os.path.exists(_meta):
            raise FileNotFoundError(
                "Result path does exist (could not find {}).".format(_meta))

        with open(_meta) as f:
            self.metadata = json.load(f)

    def data(self, keys: Optional[list[str]] = None) -> Any:
        """Get dataset file."""
        return load_arrays(self.metadata["dataset"]["path"], keys=keys)

    def map(self) -> Any:
        """Get map file."""
        return load_arrays(os.path.join(self.path, "map.mat"))

    def dart(self) -> DART:
        """Get DART object."""
        return DART.from_config(**self.metadata)

    def trajectory_dataset(  
        self, subset: Optional[Integer[Array, "nval"]] = None
    ) -> types.Dataset:
        """Load trajectory dataset."""
        return trajectory(self.metadata["dataset"]["path"], subset=subset)

    def save(self, path: str, contents: dict) -> None:
        """Save contents as a .mat file in this result scope."""
        savemat(os.path.join(self.path, path), contents)
