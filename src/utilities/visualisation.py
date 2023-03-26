# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 22, 2023
# version ='1.0'
# ---------------------------------------------------------------------------
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def plot_field(data, field, filename, xmin, xmax, ymin, ymax, nx, ny):
    x, y = np.mgrid[xmin:xmax:(nx * 1j), ymin:ymax:(ny * 1j)]

    field_idx_list = {'T': 2, 'alphaT': 3, 'k': 4, 'nut': 5, 'omega': 6,
                      'p': 7, 'rho': 8, 'Ux': 9, 'Uy': 10, 'sdf': 11}

    cbar_list = {
        2: r'Temperature $T$ [$K$]',
        3: r'Thermal Diffusivity $\alpha_T$ [$m^2/s$]',
        4: r'Turbulent Kinetic Energy $k$ [$m^2/s^2$]',
        5: r'Eddy viscosity $\nu_t$ [$m^2/s$]',
        6: r'Turbulence Specific Dissipation Rate $\omega$ [$s^{-1}$]',
        7: r'Pressure $p$ [$Pa$]',
        8: r'Density $\rho$ [$kg/m^3$]',
        9: r'Horizontal Velocity $U_x$ [$m^2/s$]',
        10: r'Vertical Velocity $U_y$ [$m^2/s$]',
        11: r'Signed Distance Field [$m$]'
    }

    field_idx = field_idx_list[field]

    if field_idx in [2, 4, 6, 7, 8]:
        min_val = np.min(np.ma.masked_values(data[:, field_idx], 0, copy=False))
    else:
        min_val = np.min(data[:, field_idx])

    max_val = data[:, field_idx].max()

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # create new axis for colorbar to be closer to figure
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "3%", pad="3%")

    im = ax.pcolormesh(x, y, data[:, field_idx].reshape(nx, ny),
                       vmin=min_val, vmax=max_val)
    plt.colorbar(im, cax=cax, label=cbar_list[field_idx],
                 ticks=np.linspace(min_val, max_val, 10, endpoint=True))

    plt.tight_layout()
    plt.savefig('../output/{}.png'.format(filename), dpi=300)
