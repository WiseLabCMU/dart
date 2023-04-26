"""Loss functions."""

from jax import numpy as jnp

from beartype.typing import Optional
from . import types


def get_loss_func(
    loss: str = "l2", weight: Optional[str] = None, eps: float = 1e-8,
    delta: float = 0.1
) -> types.LossFunc:
    """Create loss function (as closure).

    Parameters
    ----------
    loss: loss order (l1, l2 currently).
    weight: magnitude-based loss weighting (sqrt, None).
    eps: epsilon when required by sqrt, log, div, etc; ignored if unused.
    delta: huber loss parameter.

    Returns
    -------
    Loss function callable (y_pred, y_true) -> loss.
    """
    def loss_func(y_pred, y_true):
        if weight == "sqrt":
            err = jnp.sqrt(y_pred + eps) - jnp.sqrt(y_true + eps)
        elif weight == "log":
            err = jnp.log(y_pred + eps) - jnp.log(y_true + eps)
        else:
            err = y_pred - y_true

        match loss:
            case "l1":
                err = jnp.abs(err)
            case "l2":
                err = jnp.square(err)
            case "huber":
                err = jnp.where(
                    jnp.abs(err) > delta,
                    delta * (jnp.abs(err) - 0.5 * delta),
                    0.5 * jnp.square(err))
            case _:
                raise ValueError("Loss not implemented: {}".format(loss))

        return jnp.sum(err) / y_true.shape[0]

    return loss_func
