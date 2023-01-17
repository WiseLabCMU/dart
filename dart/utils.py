"""Utilities that should be included in a standard library somewhere."""

from tqdm import tqdm

import jax
import numpy as np
from jax import numpy as jnp

from jaxtyping import PyTree, Integer, Array
from beartype.typing import Union


def to_jax(batch: PyTree) -> PyTree:
    """Convert non-jax array to jax array."""
    return jax.tree_util.tree_map(jnp.array, batch)


def to_prngkey(
        key: Union[Integer[Array, "2"], int] = 42) -> Integer[Array, "2"]:
    """Accepts integer seeds or PRNGKeys."""
    if isinstance(key, int):
        return jax.random.PRNGKey(key)
    else:
        return key


def update_avg(loss: float, avg: float, idx: int, pbar: tqdm) -> float:
    """Update moving average on progress bar."""
    avg = (avg * idx + loss) / (idx + 1)
    pbar.set_postfix(loss=avg)
    return avg


def get_size(tree: PyTree) -> int:
    """Get size of pytree."""
    size, _ = jax.tree_util.tree_flatten(
        jax.tree_util.tree_map(lambda x: x.shape[0], tree))
    if not np.all(np.array(size) == size[0]):
        raise ValueError("Axis 0 of the given PyTree is not consistent.")
    else:
        return size[0]


def shuffle(
        ordered: PyTree, key: Union[Integer[Array, "2"], int] = 42) -> PyTree:
    """Shuffle arrays."""
    indices = jax.random.permutation(to_prngkey(key), get_size(ordered))
    return jax.tree_util.tree_map(lambda x: x[indices], ordered)
