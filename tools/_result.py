"""Result convenience wrapper."""

import os
import json
import numpy as np
import pandas as pd
import h5py

from beartype.typing import Any


def _json(p):
    with open(p) as f:
        return json.load(f)


class DartResult:
    """DART result/dataset convenience wrapper."""

    def __init__(self, path: str) -> None:
        _meta = os.path.join(path, "metadata.json")
        if not os.path.exists(_meta):
            raise FileNotFoundError(
                "Result path does not exist (could not find {})".format(_meta))

        self.metadata = _json(_meta)
        self.resdir = path
        self.datadir = os.path.dirname(self.metadata["dataset"]["path"])

    def dart(self) -> "DART":  # type: ignore
        """Construct DART for results.
        
        NOTE: will import DART (and load jax & other heavy dependencies) on
        first call.
        """
        from dart import DART
        return DART.from_config(**self.metadata)

    def path(self, subpath: str) -> str:
        """Translate path to result/data directory."""
        if subpath.startswith("data/"):
            return os.path.join(self.datadir, subpath.replace("data/", ""))
        else:
            return os.path.join(self.resdir, subpath.replace("result/", ""))

    def __getitem__(self, subpath: str) -> Any:
        """Load npz/csv/h5/json file.
        
        Use `data/` to indicate when to load from the dataset directory;
        otherwise, `subpath` is assumed to be in `results`. A `results/` prefix
        can also be passed (which is removed).
        """
        path = self.path(subpath)
        if not os.path.exists(path):
            raise FileNotFoundError("File does not exist: {}".format(path))

        def _err(p):
            raise ValueError("Unknown file extension: {}".format(p))

        exts = {
            ".npz": np.load, ".csv": pd.read_csv,
            ".h5": h5py.File, ".json": _json}
        return exts.get(os.path.splitext(path)[1], _err)(path)
