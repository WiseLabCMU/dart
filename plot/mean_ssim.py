"""Create (print) mean SSIM table body."""

import numpy as np
from scipy.stats import norm

from _stats import load_dir

datasets = {
    "boxes2": "Lab 1",
    "boxes3": "Lab 2",
    "wiselab4": "Office 1",
    "wiselab5": "Office 2",
    "mallesh-half": "Rowhouse 1",
    "mallesh-1br": "Rowhouse 2",
    "mallesh-full": "Rowhouse 3",
    "agr-ground": "House 1",
    "agr-full": "House 2",
    "agr-yard": "Yard",
    "tianshu-full": "Apartment 1",
    "tianshu-half": "Apartment 2"
}
methods = {
    "ngpsh": "Dart",
    "lidar": "Lidar",
    "nearest": "Nearest",
    "cfar": "CFAR",
    "ngp": "No View Dependence",
    "grid5": "20cm Grid",
    "grid10": "10cm Grid",
    "grid25": "4cm Grid"
}


agg = {k: [] for k in methods}
for ds in datasets:
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
