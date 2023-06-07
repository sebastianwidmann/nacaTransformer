# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 22, 2023
# version ='1.0'
# ---------------------------------------------------------------------------
from ml_collections import ConfigDict
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import inset_locator
import numpy as np


def plot_fields(config: ConfigDict, predictions, ground_truth, epoch):
    nrows, ncols = 3, 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(10, 15))

    xmin, xmax, ymin, ymax = config.preprocess.dim
    nx, ny = config.vit.img_size
    x, y = np.mgrid[xmin:xmax:(nx * 1j), ymin:ymax:(ny * 1j)]

    labelname = ['$p/p_\infty$', '$u_x/u_\infty$', '$u_y/u_\infty$']

    for i in range(nrows):
        z1, z2 = predictions[:, :, i], ground_truth[:, :, i]
        lower_limit, upper_limit = np.min(z2), np.max(z2)

        im0 = ax[i, 0].pcolormesh(
            x, y, z1, vmin=lower_limit, vmax=upper_limit,
        )
        im1 = ax[i, 1].pcolormesh(
            x, y, z2, vmin=lower_limit, vmax=upper_limit,
        )

        axins = inset_locator.inset_axes(ax[i, 1],
                                         width="5%", height="100%",
                                         loc='center right',
                                         borderpad=-2)
        plt.colorbar(im1, cax=axins, label=labelname[i],
                     ticks=np.linspace(lower_limit, upper_limit, 10,
                                       endpoint=True))

        for j in range(ncols):
            ax[i, j].set(adjustable='box', aspect='equal')
            ax[i, j].set_xticks(np.linspace(config.preprocess.dim[0],
                                            config.preprocess.dim[1],
                                            5, endpoint=True))
            ax[i, j].set_yticks(np.linspace(config.preprocess.dim[2],
                                            config.preprocess.dim[3],
                                            5, endpoint=True))

            plt.setp(ax[i, j].get_xticklabels(), visible=False) if i != 2 \
                else None
            plt.setp(ax[i, j].get_yticklabels(), visible=False) if j != 0 \
                else None

    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.savefig('vit_all_{}.png'.format(epoch), bbox_inches="tight", dpi=300)
    plt.close()


def plot_predictions(config: ConfigDict, predictions, ground_truth, epoch):
    xmin, xmax, ymin, ymax = config.preprocess.dim
    nx, ny = config.vit.img_size
    x, y = np.mgrid[xmin:xmax:(nx * 1j), ymin:ymax:(ny * 1j)]

    labelname = ['$p/p_\infty$', '$u_x/u_\infty$', '$u_y/u_\infty$']
    outputname = ['p', 'ux', 'uy']

    nplots, ncols = 3, 2

    for i in range(nplots):
        fig, ax = plt.subplots(1, ncols, figsize=(10, 5))

        z1, z2 = predictions[:, :, i], ground_truth[:, :, i]
        lower_limit, upper_limit = np.min(z2), np.max(z2)

        im0 = ax[0].pcolormesh(
            x, y, z1, vmin=lower_limit, vmax=upper_limit)
        im1 = ax[1].pcolormesh(
            x, y, z2, vmin=lower_limit, vmax=upper_limit)

        axins = inset_locator.inset_axes(ax[1],
                                         width="5%", height="100%",
                                         loc='center right',
                                         borderpad=-2)

        plt.colorbar(im1, cax=axins, label=labelname[i],
                     ticks=np.linspace(lower_limit, upper_limit, 10,
                                       endpoint=True))

        for j in range(ncols):
            ax[j].set(adjustable='box', aspect='equal')
            ax[j].set_xticks(np.linspace(config.preprocess.dim[0],
                                         config.preprocess.dim[1],
                                         5, endpoint=True))
            ax[j].set_yticks(np.linspace(config.preprocess.dim[2],
                                         config.preprocess.dim[3],
                                         5, endpoint=True))
            plt.setp(ax[j].get_yticklabels(), visible=False) if j != 0 else None

        plt.subplots_adjust(wspace=0.1)
        plt.savefig('vit_{}_{}.png'.format(outputname[i], epoch),
                    bbox_inches="tight", dpi=300)
        plt.close()


def plot_delta(config, predictions, ground_truth, epoch, cmap):
    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(10, 5))

    xmin, xmax, ymin, ymax = config.preprocess.dim
    nx, ny = config.vit.img_size
    x, y = np.mgrid[xmin:xmax:(nx * 1j), ymin:ymax:(ny * 1j)]

    dp = ground_truth[:, :, 0] - predictions[:, :, 0]
    dux = ground_truth[:, :, 1] - predictions[:, :, 1]
    duy = ground_truth[:, :, 2] - predictions[:, :, 2]

    def normalise(z, lower, upper):
        return (z - z.min()) / (z.max() - z.min()) * (upper - lower) + lower

    # Normalise deltas to [-1,1]
    lower_limit, upper_limit = -1, 1
    dp = normalise(dp, lower_limit, upper_limit)
    dux = normalise(dux, lower_limit, upper_limit)
    duy = normalise(duy, lower_limit, upper_limit)

    z = [dp, dux, duy]

    labelname = ['$\Delta p\'$', '$\Delta u_x\'$', '$\Delta u_y\'$']

    for i in range(3):
        axins = inset_locator.inset_axes(ax[i],
                                         width="100%", height="5%",
                                         loc='lower center',
                                         borderpad=-3)

        im = ax[i].pcolormesh(x, y, z[i],
                              cmap=cmap, vmin=lower_limit, vmax=upper_limit)
        plt.colorbar(im, cax=axins, orientation='horizontal',
                     label=labelname[i],
                     ticks=np.linspace(lower_limit, upper_limit, 5,
                                       endpoint=True))

        ax[i].set(adjustable='box', aspect='equal')
        ax[i].set_xticks(np.linspace(config.preprocess.dim[0],
                                     config.preprocess.dim[1],
                                     5, endpoint=True))
        ax[i].set_yticks(np.linspace(config.preprocess.dim[2],
                                     config.preprocess.dim[3],
                                     5, endpoint=True))
        plt.setp(ax[i].get_yticklabels(), visible=False) if i != 0 else None

    # plt.subplots_adjust(wspace=0.1)

    plt.savefig('vit_delta_{}.png'.format(epoch), bbox_inches="tight", dpi=300)
    plt.close()


def plot_loss(config: ConfigDict, train_loss, test_loss):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    x = range(0, config.num_epochs)

    ax.plot(x, train_loss, label='Train')
    ax.plot(x, test_loss, label='Test')

    ax.set_yscale('log')

    ax.set_xlim(0, config.num_epochs)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Mean Square Error')

    ax.legend(loc=1)

    plt.grid(visible=True, which='major', color='#444', linestyle='-')
    plt.grid(visible=True, which='minor', color='#ccc', linestyle='--')

    ax.set_title(
        'Epochs = {}, Batch size = {}, Lr scheduler = {}, Weight decay = {}'
        .format(config.num_epochs, config.batch_size,
                config.learning_rate_scheduler, config.weight_decay),
        fontsize=8)

    plt.savefig('vit_loss.png', bbox_inches="tight", dpi=300)
    plt.close()
