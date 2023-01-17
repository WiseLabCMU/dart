"""Train DART model."""

import json
import time
from tqdm import tqdm
from argparse import ArgumentParser

import jax

from dart import DART


def _parse():

    p = ArgumentParser()
    p.add_argument(
        "-s", "--sensor", default="data/sim_96.json",
        help="Sensor parameters (.json).")
    p.add_argument(
        "-n", default=256, type=int,
        help="Override on stochastic integration angular bin resolution (n).")
    p.add_argument(
        "-k", default=128, type=int,
        help="Override on stochastic integration number of samples (k).")

    p.add_argument("-o", "--out", default="result", help="Output file.")

    p.add_argument("-r", "--key", default=42, type=int, help="Random key.")
    p.add_argument("--lr", default=0.01, type=float, help="Learning Rate.")
    p.add_argument(
        "-e", "--epochs", default=1, type=int,
        help="Number of epochs to train.")
    p.add_argument("-b", "--batch", default=2048, type=int, help="Batch size.")

    p.add_argument("--levels", default=8, type=int, help="Hash table levels.")
    p.add_argument(
        "--exponent", default=0.5, type=float,
        help="Hash table level exponent, in powers of 2.")
    p.add_argument(
        "--base", default=4., type=float,
        help="Size of base (most coarse) hash table level.")
    p.add_argument(
        "--size", default=16, type=int,
        help="Hash table size, in powers of 2.")
    p.add_argument(
        "--features", default=2, type=int,
        help="Number of features per hash table level.")

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
    state = dart.fit(
        train.batch(cfg["batch"]), state, epochs=cfg["epochs"], tqdm=tqdm,
        key=k3, val=val.batch(cfg["batch"]))
    dart.save(cfg["out"] + ".chkpt", state)


if __name__ == '__main__':

    args = _parse().parse_args()

    with open("data/awr1843boost.json") as f:
        sensor_cfg = json.load(f)
    sensor_cfg.update(n=args.n, k=args.k)

    cfg = {
        "sensor": sensor_cfg,
        "field": {
            "levels": args.levels,
            "exponent": args.exponent,
            "base": args.base,
            "size": args.size,
            "features": args.features
        },
        "dataset": {
            "val": 0.2,
            "clip": 99.9,
            "iid_val": True,
            "path": "data/cup.mat"
        },
        "shuffle_buffer": 100 * 1000,
        "lr": args.lr,
        "batch": args.batch,
        "epochs": args.epochs,
        "key": args.key,
        "out": args.out
    }

    _main(cfg)
