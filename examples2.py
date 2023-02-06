from matplotlib import pyplot as plt
import numpy as np

from dart import dataset


sim = dataset.load_arrays("data/cabinets/simulated.mat")
obs = dataset.load_arrays("data/cabinets/cabinets.mat")
pred = dataset.load_arrays("results/cabinets_pred.mat")

def _show(ax, im):
    ax.imshow(im)

fig, axs = plt.subplots(3, 6, figsize=(16, 12))

for idx, (ax1, ax2, ax3) in zip(np.arange(8) * 654 + 300, axs.reshape(-1, 3)):
    _show(ax1, sim["rad"][idx])
    _show(ax2, obs["rad"][idx])
    _show(ax3, pred["rad"][idx])
    ax1.set_title("sim")
    ax2.set_title("obs")
    ax3.set_title("pred")

fig.savefig("results/cabinets_comparison.png")
