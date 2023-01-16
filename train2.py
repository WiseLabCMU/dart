"""Train DART model."""

import json
import time
from tqdm import tqdm
from argparse import ArgumentParser

from dart import dataset, DART


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


if __name__ == '__main__':

    args = _parse().parse_args()

    print("Setting up...")
    start = time.time()

    with open(args.sensor) as f:
        cfg = {"sensor": json.load(f)}
    cfg["sensor"].update(n=args.n, k=args.k)
    cfg.update(
        shuffle_buffer=100000, lr=args.lr, batch=args.batch,
        field={
            "levels": args.levels, "exponent": args.exponent,
            "base": args.base, "size": args.size, "features": args.features
        })

    with open(args.out + ".json", 'w') as f:
        json.dump(cfg, f)

    dart = DART.from_config(**cfg)
    ds = dataset.dart2("data/frames.mat", dart.sensor, pre_shuffle=True)
    ds = ds.shuffle(100000, reshuffle_each_iteration=True)

    print("Done setting up ({:.1f}s).".format(time.time() - start))

    state = dart.fit(
        ds.batch(args.batch), dart.init(ds.batch(2), key=args.key),
        epochs=args.epochs, tqdm=tqdm, key=args.key + 1)
    dart.save(args.out + ".chkpt", state)
