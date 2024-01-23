"""Statistical utilities."""

import os
import numpy as np
from jaxtyping import Num


BASELINES = ['lidar', 'nearest', 'cfar', 'cfar_1e-2', 'cfar_1e-5', 'cfar_1e-8']

DATASETS = {
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


def load_dir(path, key="ssim", baselines=BASELINES):
    """Load metrics from a given dataset."""

    def _try_load(*path):
        try:
            return np.load(os.path.join(*path))[key]
        except FileNotFoundError:
            return None

    res = os.path.join("results", path)
    data = os.path.join("data", path)

    contents = {k: _try_load(res, k, "metrics.npz") for k in os.listdir(res)}
    for k in baselines:
        contents[k] = _try_load(data, "baselines/{}.npz".format(k))

    return {k: v for k, v in contents.items() if v is not None}


def effective_sample_size(x: Num[np.ndarray, "t"]) -> float:
    """Calculate effective sample size for time series data."""
    rho = np.array([
        np.cov(x[i:], x[:-i])[0, 1] / np.std(x[i:]) / np.std(x[:-i])
        for i in range(1, x.shape[0] // 2)])
    rho_sum = np.sum(np.maximum(0.0, rho))
    return x.shape[0] / (1 + 2 * rho_sum)
