"""Dataset loading."""

from beartype.typing import Union, Optional
from functools import partial
from jaxtyping import Integer, Array

import jax
from jax import numpy as jnp
import numpy as np
from tensorflow.data import Dataset
from scipy.io import loadmat

from .pose import make_pose

from .utils import get_size, shuffle


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
        val: float = 0., iid_val: bool = False,
        key: Optional[Union[Integer[Array, "2"], int]] = 42
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
        val: Proportion of dataset to hold as a validation set. If val=0,
            Then no validation datset is returned.
        iid_val: If True, then shuffles the dataset before training so that the
            validation split is drawn randomly from the dataset instead of just
            from the end.
        key: Random key to shuffle dataset frames. Does not shuffle columns.

        Returns
        -------
        (train, val) datasets.
        """
        data = loadmat(path)
        data = (
            jax.vmap(make_pose)(data["vel"], data["pos"], data["rot"]),
            data["rad"] / np.percentile(data["rad"], clip))
        print("Loaded dataset: {} frames".format(data[1].shape[0]))

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
