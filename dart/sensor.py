"""Ray sampling routines.

Conventions
-----------
- The sensor field of view is centered around +x.
- +y is left of +x.
- +z is straight up.
"""

from functools import partial

from jaxtyping import Float32, Bool, Array, Integer
from beartype.typing import NamedTuple, Optional
from . import types

import numpy as np
from jax import numpy as jnp
from jax import random, vmap
from jax.tree_util import tree_map
from tensorflow.data import Dataset

from . import types
from .pose import project_angle, sensor_to_world, make_pose
from .spatial import vec_to_angle
from .antenna import antenna_gain
from .utils import get_size, shuffle
from .dataset import load_arrays


class VirtualRadar(NamedTuple):
    """Radar Sensor Model.

    Attributes
    ----------
    r, d: Range, doppler bins used for (r, d) images. Pass as (min, max, bins),
        i.e. the args of linspace in configuration (for `from_config`).
    theta_lim, phi_lim: Bounds (radians) on elevation and azimuth angle;
        +/- pi/12 (15 degrees) and pi/3 (60 degrees) by default.
    n: Angular resolution; number of bins in a full circle of the
        (range sphere, doppler plane) intersection
    k: Sample size for stochastic integration
    """

    r: Float32[Array, "Nr"]
    d: Float32[Array, "Nd"]
    theta_lim: float
    phi_lim: float
    n: int
    k: int

    @property
    def bin_width(self):
        """Alias for the width of bins in stochastic integration."""
        return 2 * jnp.pi / self.n

    @property
    def _extents(self):
        """Alias for the extents of a range-doppler plot."""
        return [self.d[0], self.d[-1], self.r[0], self.r[-1]]

    @classmethod
    def from_config(
        cls, theta_lim: float = jnp.pi / 12, phi_lim: float = jnp.pi / 3,
        n: int = 256, k: int = 128, r: list = [], d: list = []
    ) -> "VirtualRadar":
        """Create from configuration parameters."""
        return cls(
            r=jnp.linspace(*r), d=jnp.linspace(*d),
            theta_lim=theta_lim, phi_lim=phi_lim, n=n, k=k)

    def valid_mask(
        self, d: Float32[Array, ""], pose: types.RadarPose
    ) -> Bool[Array, "n"]:
        """Get valid psi values within field of view as a mask.

        Computes a mask for bins::

            jnp.arange(n) * bin_width

        Parameters
        ----------
        d: Doppler bin.
        pose: Sensor pose parameters.

        Returns
        -------
        Output mask for each bin.
        """
        t = project_angle(d, jnp.arange(self.n) * self.bin_width, pose)
        theta, phi = vec_to_angle(t)
        return (
            (theta < self.theta_lim) & (theta > -self.theta_lim)
            & (phi < self.phi_lim) & (phi > -self.phi_lim)
            & (t[0] > 0))

    def sample_rays(
            self, key, d: Float32[Array, ""],
            valid_psi: Bool[Array, "n"], pose: types.RadarPose
    ) -> Float32[Array, "3 k"]:
        """Sample rays according to pre-computed psi mask.

        Parameters
        ----------
        key: PRNGKey for random sampling.
        d: Doppler bin.
        valid_psi: Valid psi bins for angles jnp.arange(n) * bin_width in the
            (p, q) basis for the r-sphere d-plane intersection.
        pose: Sensor pose parameters.

        Returns
        -------
        Generated samples.
        """
        k1, k2 = random.split(key, 2)

        weights = valid_psi.astype(jnp.float32) * 10
        indices = random.categorical(k1, weights, shape=(self.k,))
        bin_centers = indices.astype(jnp.float32) * self.bin_width

        offsets = self.bin_width * (random.uniform(k2, shape=(self.k,)) - 0.5)
        psi_actual = bin_centers + offsets
        points = project_angle(d, psi_actual, pose)
        return points

    def render_column(
        self, t: Float32[Array, "3 k"], sigma: types.SigmaField,
        pose: types.RadarPose, weight: Float32[Array, ""]
    ) -> Float32[Array, "nr"]:
        """Render a single doppler column for a radar image.

        Parameters
        ----------
        t: Sensor-space rays on the unit sphere.
        sigma: Field function.
        pose: Sensor pose.
        weight: Sample size weight.

        Returns
        -------
        Rendered column for one doppler value and a stack of range values.
        """
        def project_rays(r):
            t_world = sensor_to_world(r=r, t=t, pose=pose)
            dx = pose.x.reshape(-1, 1) - t_world
            dx_norm = dx / jnp.linalg.norm(dx, axis=0)
            return jnp.nan_to_num(vmap(sigma)(t_world.T, dx=dx_norm.T))

        # Antenna Gain
        gain = antenna_gain(*vec_to_angle(t))

        # Field steps
        field_vals = vmap(project_rays)(self.r)
        sigma_samples = 0.001 * field_vals[:, :, 0]
        alpha_samples = 1 - 0.001 * field_vals[:, :, 1]

        # Return signal
        transmitted = jnp.concatenate([
            jnp.ones((1, t.shape[1])),
            jnp.cumprod(alpha_samples[:-1], axis=0)
        ], axis=0)
        amplitude = sigma_samples * transmitted * gain

        constant = weight / self.n * self.r
        return jnp.mean(amplitude, axis=1) * constant

    def make_column(
        self, doppler: Float32[Array, ""], pose: types.RadarPose
    ) -> types.TrainingColumn:
        """Create column for training.

        Parameters
        ----------
        d: doppler value.
        pose: sensor pose.

        Returns
        -------
        Training point with per-computed valid bins.
        """
        valid = self.valid_mask(doppler, pose)
        packed = jnp.packbits(valid)
        weight = jnp.sum(valid).astype(jnp.float32) / pose.s
        return types.TrainingColumn(
            pose=pose, valid=packed, weight=weight, doppler=doppler)

    def column_forward(
        self, key: random.PRNGKeyArray, column: types.TrainingColumn,
        sigma: types.SigmaField,
    ) -> Float32[Array, "nr"]:
        """Render a training column.

        Parameters
        ----------
        key : PRNGKey for random sampling.
        column: Pose and y_true.
        sigma: Field function.

        Returns
        -------
        Predicted doppler column.
        """
        valid = jnp.unpackbits(column.valid)
        t = self.sample_rays(
            key, d=column.doppler, valid_psi=valid, pose=column.pose)
        return self.render_column(
            t=t, sigma=sigma, pose=column.pose, weight=column.weight)

    def sample_points(
        self, key: random.PRNGKeyArray, r: Float32[Array, ""],
        d: Float32[Array, ""], pose: types.RadarPose
    ) -> tuple[Float32[Array, "3 k"], Integer[Array, ""]]:
        """Sample points in world-space for the given (range, doppler) bin.

        Parameters
        ----------
        key: PRNGKey for random sampling.
        r, d: Range and doppler bins.
        pose: Sensor pose parameters.

        Returns
        -------
        points: Sampled points in sensor space.
        num_bins: Number of occupied bins (effective weight of samples).
        """
        valid_psi = self.valid_mask(d, pose)
        num_bins = jnp.sum(valid_psi)

        points_sensor = self.sample_rays(key, d, valid_psi, pose)
        points_world = sensor_to_world(r, points_sensor, pose)
        return points_world, num_bins

    def render(
        self, key: random.PRNGKeyArray, sigma: types.SigmaField,
        pose: types.RadarPose
    ) -> Float32[Array, "nr nd"]:
        """Render single (range, doppler) radar image.

        NOTE: This function is not vmap or jit safe.

        Parameters
        ----------
        key: PRNGKey for random sampling.
        sigma: field function.
        pose: Sensor pose parameters.

        Returns
        -------
        Rendered image. Points not observed within the field of view are
        rendered as 0.
        """
        valid_psi = vmap(partial(self.valid_mask, pose=pose))(self.d)
        num_bins = jnp.sum(valid_psi, axis=1)

        keys = jnp.array(random.split(key, self.d.shape[0]))

        t_sensor = vmap(partial(self.sample_rays, pose=pose))(
            keys, d=self.d, valid_psi=valid_psi)
        return vmap(
            partial(self.render_column, sigma=sigma, pose=pose)
        )(t_sensor, weight=num_bins.astype(float)).T

    def plot_image(self, ax, image, labels=False):
        """Plot range-doppler image using matplotlib.imshow."""
        ax.imshow(
            image, extent=self._extents, aspect='auto', origin='lower')
        if labels:
            ax.set_ylabel("Range")
            ax.set_xlabel("Doppler")

    def plot_images(self, axs, images, predicted):
        """Plot predicted and actual images side by side."""
        for y_true, y_pred, ax in zip(images, predicted, axs.reshape(-1, 2)):
            self.plot_image(ax[0], y_true)
            self.plot_image(ax[1], y_pred)
            ax[0].set_title("Actual")
            ax[1].set_title("Predicted")
        ax[1].set_yticks([])

    def _make_dataset(self, data):
        """Split poses/images into columns."""
        def process_image(pose):
            return vmap(
                partial(self.make_column, pose=pose))(doppler=self.d)

        poses, images = data
        columns = vmap(process_image)(poses)
        images_col = jnp.swapaxes(images, 1, 2)
        dataset = (columns, images_col)

        # Flatten (index, doppler) order
        flattened = tree_map(lambda x: x.reshape(-1, *x.shape[2:]), dataset)

        # Remove invalid doppler columns
        not_empty = flattened[0].weight > 0
        dataset_valid = tree_map(lambda x: x[not_empty], flattened)

        return dataset_valid

    def dataset(
        self, path: str = "data/cup.mat", clip: float = 99.9,
        norm: float = 0.05, val: float = 0., iid_val: bool = False,
        min_speed: float = 0.1, key: types.PRNGSeed = 42
    ) -> tuple[Dataset, Optional[Dataset]]:
        """Real dataset trajectory and images.

        The dataset is ordered by::

            (image/pose index, doppler)

        With the image/pose shuffled. If the sensor has fewer range bins than
        are provided in the dataset, only the closest are used, and further
        bins are cropped out and removed.

        Parameters
        ----------
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
        images = images[:, :len(self.r)]

        data = vmap(make_pose)(data["vel"], data["pos"], data["rot"]), images
        valid_speed = data[0].s > min_speed

        print("Loaded dataset: {} valid frames (speed > {}) / {}".format(
            jnp.sum(valid_speed), min_speed, data[1].shape[0]))
        data = tree_map(lambda x: x[valid_speed], data)

        if iid_val:
            data = shuffle(data, key=key)

        nval = 0 if val <= 0 else int(get_size(data) * val)
        if nval > 0:
            train = tree_map(lambda x: x[:-nval], data)
            val = tree_map(lambda x: x[-nval:], data)

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
