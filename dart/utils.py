"""Utilities that should be included in a standard library somewhere."""

from tqdm import tqdm
import json

import jax
from jax import numpy as jnp
import numpy as np


from jaxtyping import PyTree
from beartype.typing import TypeVar, Optional, Union
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


def update_avg(loss: float, state: types.Average, pbar: tqdm) -> types.Average:
    """Update moving average on progress bar."""
    avg = (state.avg * state.n + loss) / (state.n + 1)
    pbar.set_postfix(loss=avg)
    return types.Average(avg=avg, n=state.n + 1)


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


def split(data: PyTree, nval: int = 0) -> tuple[PyTree, Optional[PyTree]]:
    """Create train and val splits."""
    if nval > 0:
        train = jax.tree_util.tree_map(lambda x: x[:-nval], data)
        val = jax.tree_util.tree_map(lambda x: x[-nval:], data)
        return train, val
    else:
        return data, None


def save_weights(weights: dict, path: str) -> None:
    """Save weights to the provided path."""
    # Accumulate global flattened entries
    flattened: dict[str, np.ndarray] = {}

    def _save(breadcrumb: list[str], subweights: dict) -> dict:
        subschema: dict[str, Union[str, dict]] = {}
        for k, v in subweights.items():
            subpath = "/".join(breadcrumb + [k])
            if isinstance(v, dict):
                subschema[k] = _save([subpath], v)
            else:
                flattened[subpath] = np.array(v)
                subschema[k] = subpath
        return subschema

    schema = _save([], weights)

    with open(path + ".json", 'w') as f:
        json.dump(schema, f, indent=4)
    np.savez(path + ".npz", **flattened)


def load_weights(path: str) -> dict:
    """Load weights from the provided path (.json and .npz files.)."""
    with open(path + ".json") as f:
        schema = json.load(f)
    flattened = np.load(path + ".npz")

    return jax.tree_util.tree_map(flattened.get, schema)
