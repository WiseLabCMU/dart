"""Create (print) mean SSIM table body."""

import numpy as np
from scipy.stats import norm

from _stats import load_dir, DATASETS


methods = {
    "ngpsh": "Dart",
    "lidar": "Lidar",
    "nearest": "Nearest",
    "cfar": "CFAR",
    "ngpsh2": "NeRF-style View Dependence",
    "ngp": "No View Dependence",
    "grid5": "20cm Grid",
    "grid10": "10cm Grid",
    "grid25": "4cm Grid"
}


agg = {k: [] for k in methods}
for ds in DATASETS:
    ssim = load_dir(ds)
    for k in methods:
        try:
            agg[k].append((
                np.nanmean(ssim[k]),
                np.nanmean(ssim["ngpsh"] - ssim[k])))
        except KeyError:
            agg[k].append((np.nan, np.nan))
agg = {k: np.array(v) for k, v in agg.items()}


def stderr(x):
    return np.nanstd(x, ddof=1) / np.sqrt(np.sum(~np.isnan(x)))


for k, v in methods.items():
    mu = np.nanmean(agg[k][:, 0])
    se = stderr(agg[k][:, 0]) * norm.ppf(0.95)
    mu_diff = np.nanmean(agg[k][:, 1])
    se_diff = stderr(agg[k][:, 1]) * norm.ppf(0.95)
    if mu_diff == 0.0:
        print("{} & {:.3f} $\\pm$ {:.3f} & --- \\\\".format(v, mu, se))
    else:
        print("{} & {:.3f} $\\pm$ {:.3f} & {:.3f} $\\pm$ {:.3f} \\\\".format(
            v, mu, se, mu_diff, se_diff))
