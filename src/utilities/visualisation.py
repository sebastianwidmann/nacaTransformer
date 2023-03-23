# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 22, 2023
# version ='1.0'
# ---------------------------------------------------------------------------

from matplotlib import pyplot as plt
import numpy as np


def plot_field(data, field, filename, xmin, xmax, ymin, ymax, nx, ny):
    x, y = np.mgrid[xmin:xmax:(nx * 1j), ymin:ymax:(ny * 1j)]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    field_idx_list = {'T': 2, 'alphaT': 3, 'k': 4, 'nut': 5, 'omega': 6,
                      'p': 7, 'rho': 8, 'Ux': 9, 'Uy': 10, 'sdf': 11}

    field_idx = field_idx_list[field]

    min_val = np.partition(data, 2, axis=0)[1, :][field_idx]
    max_val = data[:, field_idx].max()

    ax.pcolormesh(x, y, data[:, field_idx].reshape(nx, ny),
                  vmin=min_val, vmax=max_val)

    plt.tight_layout()

    # plt.show()

    plt.savefig('../{}.png'.format(filename), dpi=300)
