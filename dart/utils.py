"""Utilities that should be included in a standard library somewhere."""

from tqdm import tqdm

import jax
from jax import numpy as jnp
import numpy as np


from jaxtyping import PyTree
from beartype.typing import TypeVar
from . import types


def tf_to_jax(batch: PyTree) -> PyTree:
    """Convert tensorflow array to jax array without copying."""
    return jax.tree_util.tree_map(jnp.array, batch)


def to_prngkey(key: types.PRNGSeed = 42) -> jax.random.PRNGKeyArray:
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


T = TypeVar("T")


def shuffle(ordered: T, key: types.PRNGSeed = 42) -> T:
    """Shuffle arrays."""
    indices = jax.random.permutation(to_prngkey(key), get_size(ordered))
    return jax.tree_util.tree_map(lambda x: x[indices], ordered)
