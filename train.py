"""Train DART model."""

import json
import time
from tqdm import tqdm
import os
from argparse import ArgumentParser

import jax

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
        "--exponent", default=0.43, type=float,
        help="Hash table level exponent, in powers of 2.")
    g.add_argument(
        "--base", default=10., type=float,
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
    g.add_argument(
        "--pval", default=0.2, type=float,
        help="Validation data holdout proportion.")
    g.add_argument("--loss", default="l2", help="Loss function.")
    g.add_argument("--weight", default=None, help="Loss weighting.")

    g = p.add_argument_group(title="Dataset")
    g.add_argument(
        "--norm", default=1.0, type=float,
        help="Percentile normalization value.")
    g.add_argument(
        "--min_speed", default=0.1, type=float, help="Reject frames with "
        "insufficient (i.e. not enough doppler bins); 0 to disable.")
    g.add_argument(
        "-p", "--path", default="data/cup.mat", help="Dataset file.")
    g.add_argument(
        "--device", default=0, type=int,
        help="Device index to use for computation (default 0).")
    g.add_argument(
        "--repeat", default=0, type=int,
        help="Repeat dataset within each epoch to cut down on overhead.")

    return p


def _main(cfg):

    os.makedirs(cfg["out"], exist_ok=True)
    print(json.dumps(cfg))
    print("Setting up...")
    start = time.time()

    root = jax.random.PRNGKey(cfg["key"])
    k1, k2, k3 = jax.random.split(root, 3)

    dart = DART.from_config(**cfg)
    train, val = dataset.doppler_columns(dart.sensor, key=k1, **cfg["dataset"])
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


if __name__ == '__main__':

    args = _parse().parse_args()
    jax.default_device(jax.devices("gpu")[args.device])

    with open(args.sensor) as f:
        sensor_cfg = json.load(f)
    sensor_cfg.update(n=args.n, k=args.k)

    cfg = {
        "sensor": sensor_cfg,
        "shuffle_buffer": 200 * 1000, "lr": args.lr, "batch": args.batch,
        "epochs": args.epochs, "key": args.key, "out": args.out,
        "loss": {
            "weight": args.weight, "loss": args.loss, "eps": 1e-6
        },
        "field": {
            "levels": args.levels, "exponent": args.exponent,
            "base": args.base, "size": args.size, "features": args.features
        },
        "dataset": {
            "pval": args.pval, "norm": args.norm,
            "iid_val": True, "path": args.path, "min_speed": args.min_speed,
            "repeat": args.repeat
        }
    }

    _main(cfg)
