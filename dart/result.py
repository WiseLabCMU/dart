"""Result convenience wrapper."""

import os
import json
import h5py

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
from jax import numpy as jnp

from beartype.typing import Optional, Any
from jaxtyping import Integer, Array, Float, UInt8

from .dart import DART
from .utils import colormap
from .dataset import load_arrays, trajectory
from . import types


class DartResult:
    """DART experiment results.

    Parameters
    ----------
    path: result path.
    """

    METADATA = "metadata.json"
    VALSET = "metadata.npz"
    MAP = "map.h5"
    WEIGHTS = "model.npz"
    CAMERA = "cam.h5"
    RADAR = "rad.h5"

    def __init__(self, path: str, name: Optional[str] = None) -> None:
        self.path = path
        self.name = path if name is None else name

        _meta = os.path.join(self.path, self.METADATA)
        if not os.path.exists(_meta):
            raise FileNotFoundError(
                "Result path does exist (could not find {}).".format(_meta))

        with open(_meta) as f:
            self.metadata = json.load(f)

        self.DATASET = self.metadata["dataset"]["path"]

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

    def open(self, path: str) -> Any:
        """Load h5py file in this scope."""
        return h5py.File(os.path.join(self.path, path))

    @staticmethod
    def colorize_map(
        arr: Float[types.ArrayLike, "..."], sigma: bool = True,
        clip: tuple[float, float] = (1.0, 99.0), conv: int = 0
    ) -> UInt8[types.ArrayLike, "... 3"]:
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

    @staticmethod
    def colorize_radar(
        cmap: Float[types.ArrayLike, "..."],
        rad: Float[types.ArrayLike, "..."],
        clip: tuple[float, float] = (5.0, 99.9)
    ) -> UInt8[types.ArrayLike, "... 3"]:
        """Colorize a radar intensity map in a jax-friendly way.

        Parameters
        ----------
        cmap: color map to apply.
        rad: input array. If range-doppler, colorize directly; if
            range-doppler-azimuth, tiles into 2 columns x 4 rows.
        clip: percentile clipping range.
        """
        def _tile(arr):
            unpack = [arr[:, :, :, i] for i in range(arr.shape[3])]
            left = jnp.concatenate(unpack[:4], axis=1)
            right = jnp.concatenate(unpack[4:], axis=1)
            return jnp.concatenate([left, right], axis=2)

        p5, p95 = jnp.nanpercentile(rad, jnp.array(clip))
        rad = (rad - p5) / (p95 - p5)
        colors = (colormap(cmap, rad) * 255).astype(jnp.uint8)
        if len(rad.shape) == 4:
            if rad.shape[-1] > 1:
                return _tile(colors)
            else:
                return colors[..., 0, :]
        else:
            return colors

    def plot_map(
        self, fig, ax, layer: int = 50, checkpoint: str = "map.h5",
        key: str = "sigma", trajectory: bool = True,
        clip: tuple[float, float] = (1.0, 99.0), filter: int = -1
    ) -> None:
        """Draw map."""
        mapfile = self.load(checkpoint)

        if filter <= 0:
            layer = mapfile[key][:, :, layer]
        else:
            layer = np.median(
                mapfile[key][:, :, layer - filter:layer + filter], axis=2)

        if key == "alpha":
            layer = np.exp(layer)
        else:
            lower, upper = np.percentile(layer, clip)
            layer = np.clip(layer, max(lower, 0.0), upper)

        xmin, ymin, zmin = mapfile["lower"].reshape(-1)
        xmax, ymax, zmax = mapfile["upper"].reshape(-1)
        extents = [xmin, xmax, ymin, ymax]

        ims = ax.imshow(np.rot90(layer, k=1), extent=extents, aspect='equal')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(ims, cax)

        if trajectory:
            traj = self.data(["pos"])["pos"]
            ax.plot(traj[:, 0], traj[:, 1], color='red', linewidth=0.5)

        ax.set_title(self.name)
        ax.grid()
