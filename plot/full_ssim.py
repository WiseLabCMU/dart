"""Create (print) full SSIM table body."""

import numpy as np
from scipy.stats import norm

from _stats import load_dir, effective_sample_size


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
methods = ["lidar", "nearest", "cfar"]


print(r"Dataset & DART & Lidar & Nearest & CFAR \\ ")
for ds, label in datasets.items():
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
