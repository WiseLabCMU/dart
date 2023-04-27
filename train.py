"""Train DART model."""

import json
from argparse import ArgumentParser

import jax

from dart import train_dart
from dart import fields


def _parse_common(p: ArgumentParser) -> ArgumentParser:

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
    g.add_argument(
        "--device", default=0, type=int,
        help="GPU index to use for computation (default 0).")

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
        "--repeat", default=0, type=int,
        help="Repeat dataset within each epoch to cut down on overhead.")

    return p


if __name__ == '__main__':

    parser = ArgumentParser(
        description="Train Doppler-Aided Radar Tomography Model.")
    subparsers = parser.add_subparsers()
    for k, v in fields._fields.items():
        p = subparsers.add_parser(
            k, help=v._description, description=v._description)
        _parse_common(p)
        v.to_parser(p.add_argument_group("Field"))
        p.set_defaults(field=v)
    args = parser.parse_args()

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
        "dataset": {
            "pval": args.pval, "norm": args.norm, "iid_val": True,
            "path": args.path, "min_speed": args.min_speed,
            "repeat": args.repeat
        }
    }
    cfg.update(args.field.args_to_config(args))
    train_dart(cfg)
