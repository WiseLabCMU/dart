"""DART Training Script."""

import json
import time
import os
from tqdm import tqdm

import numpy as np
import jax

from .dart import DART
from .dataset import doppler_columns


def script_train(cfg: dict) -> None:
    """Train DART model from configuration dictionary."""
    os.makedirs(cfg["out"], exist_ok=True)
    print(json.dumps(cfg))
    print("Setting up...")
    start = time.time()

    root = jax.random.PRNGKey(cfg["key"])
    k1, k2, k3 = jax.random.split(root, 3)

    dart = DART.from_config(**cfg)
    train, val, validx = doppler_columns(dart.sensor, key=k1, **cfg["dataset"])
    assert val is not None
    train = train.shuffle(cfg["shuffle_buffer"], reshuffle_each_iteration=True)

    print("Done setting up ({:.1f}s).".format(time.time() - start))

    state = dart.init(train.batch(2), key=k2)
    state, train_log, val_log = dart.fit(
        train.batch(cfg["batch"]), state, epochs=cfg["epochs"], tqdm=tqdm,
        key=k3, val=val.batch(cfg["batch"]))
    dart.save(os.path.join(cfg["out"], "model.chkpt"), state)

    cfg["train_log"] = train_log
    cfg["val_log"] = val_log
    with open(os.path.join(cfg["out"], "metadata.json"), 'w') as f:
        json.dump(cfg, f, indent=4)

    np.savez(os.path.join(cfg["out"], "metadata.npz"), validx=validx)
