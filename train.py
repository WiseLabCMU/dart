"""Train DART model."""

import os
import json
from argparse import ArgumentParser

import jax

from dart import fields, script_train


def _parse_common(p: ArgumentParser) -> ArgumentParser:

    p.add_argument("-o", "--out", default="result", help="Output directory.")
    p.add_argument(
        "-v", "--overwrite", action='store_true', default=False,
        help="Allow overwriting existing result.")
    p.add_argument("-r", "--key", default=42, type=int, help="Random key.")
    p.add_argument(
        "--device", default=0, type=int,
        help="GPU index to use for computation (default 0).")

    g = p.add_argument_group(title="Sensor")
    g.add_argument("--sensor", default=None, help="Sensor parameters (.json).")
    g.add_argument(
        "-k", default=128, type=int,
        help="Override on stochastic integration number of samples (k).")

    g = p.add_argument_group(title="Training")
    g.add_argument("--lr", default=0.01, type=float, help="Learning Rate.")
    g.add_argument(
        "-e", "--epochs", default=5, type=int,
        help="Number of epochs to train.")
    g.add_argument("-b", "--batch", default=1024, type=int, help="Batch size.")
    g.add_argument(
        "--pval", default=0.2, type=float,
        help="Validation data holdout proportion.")
    g.add_argument(
        "-i", "--iid", default=False, action='store_true',
        help="Use IID validation split.")
    g.add_argument("--loss", default="l1", help="Loss function.")
    g.add_argument(
        "--loss_delta", default=1.0, type=float,
        help="Delta for huber loss.")
    g.add_argument("--weight", default=None, help="Loss weighting.")
    g.add_argument(
        "--adj", type=float, default=-1, help="Adjustment regularization.")

    g = p.add_argument_group(title="Dataset")
    g.add_argument(
        "--norm", default=1.0, type=float,
        help="Normalization value.")
    g.add_argument(
        "--min_speed", default=0.2, type=float, help="Reject frames with "
        "insufficient (i.e. not enough doppler bins); 0 to disable.")
    g.add_argument(
        "-p", "--path", default="data", help="Dataset file or directory.")
    g.add_argument(
        "--repeat", default=0, type=int,
        help="Repeat dataset within each epoch to cut down on overhead.")
    g.add_argument(
        "--decimate", default=0, type=int,
        help="Apply a doppler decimation factor via stochastic averaging. "
        "Must be a multiple of the number of doppler bins.")

    return p


if __name__ == '__main__':

    parser = ArgumentParser(
        description="Train Doppler-Aided Radar Tomography Model.")
    subparsers = parser.add_subparsers()
    for k, v in fields._fields.items():
        desc = "Train {} (fields.{}).".format(v._description, v.__name__)
        p = subparsers.add_parser(k, help=desc, description=desc)
        _parse_common(p)
        v.to_parser(p.add_argument_group("Field"))
        p.set_defaults(field=v)
    args = parser.parse_args()

    if os.path.exists(args.out) and not args.overwrite:
        print("Output path already exists!")
        print("Use -v (--overwrite) if this is intended.")
        exit(-1)

    # Directory input -> use default sensor, dataset name.
    if os.path.isdir(args.path):
        if args.sensor is None:
            args.sensor = os.path.join(args.path, "sensor.json")
        args.path = os.path.join(args.path, "data.h5")

    jax.default_device(jax.devices("gpu")[args.device])

    with open(args.sensor) as f:
        sensor_cfg = json.load(f)
    sensor_cfg.update(k=args.k)

    cfg = {
        "sensor": sensor_cfg,
        "shuffle_buffer": 500 * 1000, "lr": args.lr, "batch": args.batch,
        "epochs": args.epochs, "key": args.key, "out": args.out,
        "loss": {
            "weight": args.weight, "loss": args.loss, "eps": 1e-6,
            "delta": args.loss_delta
        },
        "dataset": {
            "pval": args.pval, "norm": args.norm, "iid_val": args.iid,
            "path": args.path, "min_speed": args.min_speed,
            "repeat": args.repeat, "doppler_decimation": args.decimate
        },
        "schedules": {}
    }

    if args.adj < 0:
        cfg["adjustment_name"] = "Identity"
        cfg["adjustment"] = {}
    else:
        cfg["adjustment_name"] = "Position"
        cfg["adjustment"] = {"n": 50, "k": 200, "alpha": args.adj}

    cfg.update(args.field.args_to_config(args))
    script_train(cfg)
