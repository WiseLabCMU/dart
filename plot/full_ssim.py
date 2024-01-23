"""Create (print) full SSIM table body."""

import numpy as np
from scipy.stats import norm

from _stats import load_dir, effective_sample_size, DATASETS


def print_table(methods):
    for ds, label in DATASETS.items():
        row = [label]
        ssim = load_dir(ds)
        row += ["{:.3f}".format(np.nanmean(ssim["ngpsh"]))]
        for k in methods:
            try:
                v = ssim[k]
                diff = ssim["ngpsh"] - v

                ess = effective_sample_size(diff)
                row += [
                    r"{:.3f} ({:.3f} $\pm$ {:.3f})".format(
                        np.nanmean(v), np.nanmean(diff),
                        norm.ppf(0.95) * np.nanstd(diff, ddof=1) / np.sqrt(ess))
                ]
            except KeyError:
                row += [""]

        print(" & ".join(row) + r" \\")

print("\nmain")
print_table(["lidar", "nearest", "cfar"])
print("\ngrid")
print_table(["ngp", "grid5", "grid10", "grid25"])
print("\nngpsh2")
print_table(["ngp", "ngpsh2"])
