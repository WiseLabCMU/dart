"""Train DART model."""

import json
import time
from tqdm import tqdm
from argparse import ArgumentParser

import jax
import numpy as np
from jax import numpy as jnp
from matplotlib import pyplot as plt

from dart import DART, dataset


def _parse():

    p = ArgumentParser()

    p.add_argument("-o", "--out", default="result", help="Output base name.")

    g = p.add_argument_group(title="Sensor")
    g.add_argument(
        "-s", "--sensor", default="data/awr1843boost-cup.json",
        help="Sensor parameters (.json).")
    g.add_argument(
        "-n", default=256, type=int,
        help="Override on stochastic integration angular bin resolution (n).")
    g.add_argument(
        "-k", default=128, type=int,
        help="Override on stochastic integration number of samples (k).")

    g = p.add_argument_group(title="Field")
    g.add_argument("--levels", default=8, type=int, help="Hash table levels.")
    g.add_argument(
        "--exponent", default=0.4, type=float,
        help="Hash table level exponent, in powers of 2.")
    g.add_argument(
        "--base", default=4., type=float,
        help="Size of base (most coarse) hash table level.")
    g.add_argument(
        "--size", default=16, type=int,
        help="Hash table size, in powers of 2.")
    g.add_argument(
        "--features", default=2, type=int,
        help="Number of features per hash table level.")

    g = p.add_argument_group(title="Training")
    g.add_argument("-r", "--key", default=42, type=int, help="Random key.")
    g.add_argument("--lr", default=0.01, type=float, help="Learning Rate.")
    g.add_argument(
        "-e", "--epochs", default=1, type=int,
        help="Number of epochs to train.")
    g.add_argument("-b", "--batch", default=2048, type=int, help="Batch size.")
    g.add_argument("--val", default=0.2, type=float, help="Validation size.")

    g = p.add_argument_group(title="Dataset")
    g.add_argument(
        "--clip", default=0.0, type=float,
        help="Percentile to normalize input values by.")
    g.add_argument(
        "-p", "--path", default="data/cup.mat", help="Dataset file.")

    return p


def _main(cfg):

    print(json.dumps(cfg))
    print("Setting up...")
    start = time.time()

    root = jax.random.PRNGKey(cfg["key"])
    k1, k2, k3 = jax.random.split(root, 3)

    dart = DART.from_config(**cfg)
    train, val = dart.sensor.dataset(key=k1, **cfg["dataset"])
    train = train.shuffle(cfg["shuffle_buffer"], reshuffle_each_iteration=True)

    print("Done setting up ({:.1f}s).".format(time.time() - start))

    state = dart.init(train.batch(2), key=k2)
    state, train_log, val_log = dart.fit(
        train.batch(cfg["batch"]), state, epochs=cfg["epochs"], tqdm=tqdm,
        key=k3, val=val.batch(cfg["batch"]))
    dart.save(cfg["out"] + ".chkpt", state)

    cfg["train_log"] = train_log
    cfg["val_log"] = val_log
    with open(cfg["out"] + ".json", 'w') as f:
        json.dump(cfg, f, indent=4)

    # Grid
    """
    x = jnp.linspace(-3, 3, 100)
    grid = dart.grid(state, x, x, x)
    np.savez_compressed(cfg["out"] + ".npz", grid=grid)

    # Sample images
    traj = dataset.image_traj(cfg["dataset"]["path"])
    poses, images = list(traj.shuffle(10000).batch(12).take(1))[0]

    poses_jnp = jax.tree_util.tree_map(jnp.array, poses)
    predicted = dart.render(state, poses_jnp, key=42)
    fig, axs = plt.subplots(2, 10, figsize=(24, 16))
    dart.sensor.plot_images(axs, images, predicted)
    fig.tight_layout()

    fig.savefig(cfg["out"] + ".png")
    """


if __name__ == '__main__':

    args = _parse().parse_args()

    with open(args.sensor) as f:
        sensor_cfg = json.load(f)
    sensor_cfg.update(n=args.n, k=args.k)

    cfg = {
        "sensor": sensor_cfg,
        "shuffle_buffer": 100 * 1000, "lr": args.lr, "batch": args.batch,
        "epochs": args.epochs, "key": args.key, "out": args.out,
        "field": {
            "levels": args.levels, "exponent": args.exponent,
            "base": args.base, "size": args.size,
            "features": args.features
        },
        "dataset": {
            "val": args.val, "clip": args.clip,
            "iid_val": True, "path": args.path
        }
    }

    _main(cfg)
