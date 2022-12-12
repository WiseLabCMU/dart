"""Generate simulated data."""

from tqdm import tqdm
from argparse import ArgumentParser
from functools import partial

import numpy as np
from jax import numpy as jnp
import jax

from dart import VirtualRadar, dataset


def _parse():

    p = ArgumentParser()
    p.add_argument(
        "-c", "--rmin", default=0., type=float, help="Minimum range.")
    p.add_argument(
        "-f", "--rmax", default=10., type=float, help="Maximum range.")
    p.add_argument(
        "-l", "--dmin", default=-5., type=float, help="Minimum doppler.")
    p.add_argument(
        "-r", "--dmax", default=5, type=float, help="Maximum doppler.")
    p.add_argument(
        "-v", "--rres", default=64, type=int, help="Range bins.")
    p.add_argument(
        "-w", "--dres", default=64, type=int, help="Doppler bins.")
    p.add_argument(
        "-p", "--phi", default=jnp.pi / 3, type=float,
        help="Azimuth field of view.")
    p.add_argument(
        "-t", "--theta", default=jnp.pi / 12, type=float,
        help="Elevation field of view.")
    p.add_argument(
        "-n", default=256, type=int, help="Number of angular bins.")
    p.add_argument(
        "-k", default=128, type=int, help="Number of sampled rays.")
    p.add_argument(
        "-s", "--seed", default=42, type=int, help="Random seed.")
    p.add_argument("-o", "--out", default="simulated", help="Save base path.")
    p.add_argument(
        "-g", "--gt", default="data/map.mat", help="Ground truth reflectance.")
    p.add_argument(
        "-j", "--traj", default="data/traj.mat", help="Sensor trajectory.")
    p.add_argument("-b", "--batch", default=64, type=int, help="Batch size")
    p.add_argument("--from", default=None, )
    return p


if __name__ == '__main__':

    args = _parse().parse_args()

    sensor = VirtualRadar(
        r=jnp.linspace(args.rmin, args.rmax, args.rres),
        d=jnp.linspace(args.dmin, args.dmax, args.dres),
        phi_lim=args.phi, theta_lim=args.theta, n=args.n, k=args.k)
    sensor.to_config(args.out + ".json")

    gt = dataset.gt_map(args.gt)
    traj = dataset.trajectory(args.traj)

    render = partial(sensor.render, sigma=gt)
    render = jax.jit(jax.vmap(render))

    root_key = jax.random.PRNGKey(args.seed)

    frames = []
    for batch in tqdm(traj.batch(args.batch)):
        root_key, key = jax.random.split(root_key, 2)
        pose = jax.tree_util.tree_map(jnp.array, batch)
        keys = jnp.array(jax.random.split(key, batch.x.shape[0]))

        frames.append(np.asarray(render(pose=pose, key=keys)))

    frames = np.concatenate(frames, axis=0)
    np.savez_compressed(args.out + ".npz", y=frames)
