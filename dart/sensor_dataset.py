"""Dataset loading."""

from beartype.typing import Optional
from functools import partial
from . import types

import jax
from jax import numpy as jnp
import numpy as np
from tensorflow.data import Dataset

from .pose import make_pose
from .utils import get_size, shuffle
from .dataset import load_arrays


class VirtualRadarDatasetMixins:
    """Dataset loading methods."""

    def _make_dataset(self, data):
        """Split poses/images into columns."""
        def process_image(pose):
            return jax.vmap(
                partial(self.make_column, pose=pose))(doppler=self.d)

        poses, images = data
        columns = jax.vmap(process_image)(poses)
        images_col = jnp.swapaxes(images, 1, 2)
        dataset = (columns, images_col)

        # Flatten (index, doppler) order
        flattened = jax.tree_util.tree_map(
            lambda x: x.reshape(-1, *x.shape[2:]), dataset)

        # Remove invalid doppler columns
        not_empty = flattened[0].weight > 0
        dataset_valid = jax.tree_util.tree_map(
            lambda x: x[not_empty], flattened)

        return dataset_valid

    def dataset(
        self, path: str = "data/cup.mat", clip: float = 99.9,
        norm: float = 0.05, val: float = 0., iid_val: bool = False,
        min_speed: float = 0.1, key: types.PRNGSeed = 42
    ) -> tuple[Dataset, Optional[Dataset]]:
        """Real dataset with all in one.

        The dataset is ordered by::

            (image/pose index, doppler)

        With the image/pose shuffled.

        Parameters
        ----------
        sensor: Sensor specifications.
        path: Path to file containing data.
        clip: Percentile to normalize input values by.
        norm: Normalization factor.
        val: Proportion of dataset to hold as a validation set. If val=0,
            Then no validation datset is returned.
        iid_val: If True, then shuffles the dataset before training so that the
            validation split is drawn randomly from the dataset instead of just
            from the end.
        min_speed: Minimum speed for usable samples. Images with lower
            velocities are rejected.
        key: Random key to shuffle dataset frames. Does not shuffle columns.

        Returns
        -------
        (train, val) datasets.
        """
        data = load_arrays(path)
        images = data["rad"]
        if clip > 0:
            images = images / np.percentile(images, clip) * norm

        data = (
            jax.vmap(make_pose)(data["vel"], data["pos"], data["rot"]),
            images)
        valid_speed = data[0].s > min_speed

        print("Loaded dataset: {} valid frames (speed > {}) / {}".format(
            jnp.sum(valid_speed), min_speed, data[1].shape[0]))
        data = jax.tree_util.tree_map(lambda x: x[valid_speed], data)

        if iid_val:
            data = shuffle(data, key=key)

        nval = 0 if val <= 0 else int(get_size(data) * val)
        if nval > 0:
            train = jax.tree_util.tree_map(lambda x: x[:-nval], data)
            val = jax.tree_util.tree_map(lambda x: x[-nval:], data)

            val = self._make_dataset(val)
            print("Test split  : {} images --> {} valid columns".format(
                nval, val[1].shape))
            valset = Dataset.from_tensor_slices(val)
        else:
            train = data
            valset = None

        if not iid_val:
            train = shuffle(train, key=key)

        train = self._make_dataset(train)
        trainset = Dataset.from_tensor_slices(train)
        print("Train split : {} images --> {} valid columns".format(
            data[1].shape[0] - int(nval), train[1].shape))

        return trainset, valset
