import h5py
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import ndimage


f = h5py.File("data/cichall-3/data.h5")
start = 100
end = -1
step = 1

# f = h5py.File("data/cichall-az8-f16/data.mat")
rad = f['rad'][start:end:step]
timestamps = f['t'][start:end:step] - f['t'][0]
vel = f['vel'][start:end:step]
speed_est = f['speed'][start:end:step]

speed = np.linalg.norm(vel, axis=1)

DMAX = 1.89
left = np.maximum(0, (1 - speed / DMAX) * 256)
lefts = np.maximum(0, (1 - speed_est / DMAX) * 256)
right = np.minimum(512, (speed / DMAX) * 256 + 256)
rights = np.minimum(512, (speed_est / DMAX) * 256 + 256)

plt.ion()
fig, axs = plt.subplots(4, 2, figsize=(20, 20))
for idx in range(rad.shape[0]):
    for ax in axs.reshape(-1):
        ax.clear()

    for i, ax in enumerate(axs.reshape(-1)):
        ax.imshow(rad[idx][..., i])
        ax.axvline(left[idx], color='C0')
        ax.axvline(right[idx], color='C0')
        ax.axvline(lefts[idx], color='C1')
        ax.axvline(rights[idx], color='C1')

    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.suptitle("{} @ {:.3f}".format(idx, float(timestamps[idx])))
