from matplotlib import pyplot as plt
import numpy as np

from dart import dataset


sim = dataset.load_arrays("data/cubes/simulated.mat")
# obs = dataset.load_arrays("data/cabinets/cabinets.mat")
pred = dataset.load_arrays("results/cubes_pred.mat")

def _show(ax, im):
    ax.imshow(im)

fig, axs = plt.subplots(4, 4, figsize=(16, 16))

for idx, (ax1, ax2) in zip(np.arange(8) * 600, axs.reshape(-1, 2)):
    _show(ax1, sim["rad"][idx])
    # _show(ax2, obs["rad"][idx])
    _show(ax2, pred["rad"][idx])
    ax1.set_title("sim")
    # ax2.set_title("obs")
    ax2.set_title("pred")

fig.savefig("results/cubes_comparison.png")
