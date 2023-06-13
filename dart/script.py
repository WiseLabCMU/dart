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
    print("Setting up...")
    os.makedirs(cfg["out"], exist_ok=True)
    print(json.dumps(cfg))

    start = time.time()

    root = jax.random.PRNGKey(cfg["key"])
    k1, k2, k3 = jax.random.split(root, 3)

    dart = DART.from_config(**cfg)
    train, val, validx = doppler_columns(dart.sensor, key=k1, **cfg["dataset"])
    assert val is not None
    train = train.shuffle(cfg["shuffle_buffer"], reshuffle_each_iteration=True)

    setup_time = time.time() - start
    print("Done setting up ({:.1f}s).".format(setup_time))

    print("Training...")
    start = time.time()

    state = dart.init(train.batch(2), key=k2)
    state, train_log, val_log = dart.fit(
        train.batch(cfg["batch"]), state, epochs=cfg["epochs"], tqdm=tqdm,
        key=k3, val=val.batch(cfg["batch"]),
        save=os.path.join(cfg["out"], "checkpoints", "checkpoint"))
    dart.save(os.path.join(cfg["out"], "model"), state)

    train_time = time.time() - start
    print("Done training ({:.1f}s).".format(train_time))

    cfg["time"] = {"setup": setup_time, "train": train_time}
    cfg["train_log"] = train_log
    cfg["val_log"] = val_log
    with open(os.path.join(cfg["out"], "metadata.json"), 'w') as f:
        json.dump(cfg, f, indent=4)

    np.savez(os.path.join(cfg["out"], "metadata.npz"), validx=validx)
