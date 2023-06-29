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


def plot_fields(config: ConfigDict, predictions, ground_truth, epoch, idx):
    nrows, ncols = 3, 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(10, 15))

    xmin, xmax, ymin, ymax = config.preprocess.dim
    nx, ny = config.vit.img_size
    x, y = np.mgrid[xmin:xmax:(nx * 1j), ymin:ymax:(ny * 1j)]

    labelname = ['$p/p_\infty$', '$u_x/u_\infty$', '$u_y/u_\infty$']

    for i in range(nrows):
        z1, z2 = predictions[:, :, i], ground_truth[:, :, i]

        if i == 0:
            lower_limit, upper_limit = np.min(z2[np.nonzero(z2)]), np.max(z2)
        else:
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
    plt.savefig('{}/vit_all_{}_{}.png'.format(config.output_dir, epoch, idx),
                bbox_inches="tight", dpi=300)
    plt.close()


def plot_predictions(config: ConfigDict, predictions, ground_truth, epoch, idx):
    xmin, xmax, ymin, ymax = config.preprocess.dim
    nx, ny = config.vit.img_size
    x, y = np.mgrid[xmin:xmax:(nx * 1j), ymin:ymax:(ny * 1j)]

    labelname = ['$p/p_\infty$', '$u_x/u_\infty$', '$u_y/u_\infty$']
    outputname = ['p', 'ux', 'uy']

    nplots, ncols = 3, 2

    for i in range(nplots):
        fig, ax = plt.subplots(1, ncols, figsize=(10, 5))

        z1, z2 = predictions[:, :, i], ground_truth[:, :, i]

        if i == 0:
            lower_limit, upper_limit = np.min(z2[np.nonzero(z2)]), np.max(z2)
        else:
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
        plt.savefig('{}/vit_{}_{}_{}.png'.format(config.output_dir,
                                                 outputname[i], epoch, idx),
                    bbox_inches="tight", dpi=300)
        plt.close()


def plot_delta(config, predictions, ground_truth, epoch, i, cmap='cividis'):
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

    plt.savefig('{}/vit_delta_{}_{}.png'.format(config.output_dir, epoch, i),
                bbox_inches="tight", dpi=300)
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

    plt.savefig('{}/vit_loss.png'.format(config.output_dir),
                bbox_inches="tight",
                dpi=300)
    plt.close()


def loss_comparison(files, labels, title, hyperparameter):
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))

    for i in range(len(files)):
        data = np.loadtxt(files[i], delimiter=',')

        ax[0].plot(range(0, data.shape[0]), data[:, 0], label=labels[i])
        ax[1].plot(range(0, data.shape[0]), data[:, 1], label=labels[i])

    for j in range(2):
        ax[j].set_yscale('log')
        ax[j].set_xlim(0, 50)
        ax[j].set_ylim(1e-5, 1e-2)
        ax[j].set_xlabel('Epochs')

        # plt.setp(ax[j].get_yticklabels(), visible=False) if j != 0 else None

        ax[j].grid(visible=True, which='major', color='#444', linestyle='-')
        ax[j].grid(visible=True, which='minor', color='#ccc',
                   linestyle='--')

    ax[0].legend(loc=1, ncols=2, title=title)
    ax[0].set_ylabel('Train Loss (MSE)')
    ax[1].set_ylabel('Test Loss (MSE)')

    plt.tight_layout()

    plt.savefig('vit_loss_{}.png'.format(hyperparameter),
                bbox_inches="tight", dpi=300)
    plt.close()


def plot_preprocess(config: ConfigDict, encoder, decoder):
    xmin, xmax, ymin, ymax = config.preprocess.dim
    nx, ny = config.vit.img_size
    x, y = np.mgrid[xmin:xmax:(nx * 1j), ymin:ymax:(ny * 1j)]

    labelname = ['Mach $M$', '$p/p_\infty$', '$u_x/u_\infty$', '$u_y/u_\infty$']
    outputname = ['p', 'ux', 'uy']

    z0 = encoder[:, :, 0]
    z1 = decoder[:, :, 0]
    z2 = decoder[:, :, 1]
    z3 = decoder[:, :, 2]

    z = [z0, z1, z2, z3]

    nrows, ncols = 2, 2

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 10))

    im0 = ax[0, 0].pcolormesh(
        x, y, z0, vmin=z0.min(), vmax=z0.max())

    im1 = ax[0, 1].pcolormesh(
        x, y, z1, vmin=np.min(z1[np.nonzero(z1)]), vmax=z1.max())

    im2 = ax[1, 0].pcolormesh(
        x, y, z2, vmin=z2.min(), vmax=z2.max())

    im3 = ax[1, 1].pcolormesh(
        x, y, z3, vmin=z3.min(), vmax=z3.max())

    im = [im0, im1, im2, im3]

    idx = [(0, 0), (0, 1), (1, 0), (1, 1)]
    ticks = [np.linspace(z0.min(), z0.max(), 2, endpoint=True),
             np.linspace(np.min(z1[np.nonzero(z1)]), z1.max(), 10,
                         endpoint=True),
             np.linspace(z2.min(), z2.max(), 10, endpoint=True),
             np.linspace(z3.min(), z3.max(), 10, endpoint=True)]

    for k in range(4):
        axins = inset_locator.inset_axes(ax[idx[k]],
                                         width="5%", height="100%",
                                         loc='center right',
                                         borderpad=-2)

        plt.colorbar(im[k], cax=axins, label=labelname[k], ticks=ticks[k])

        # ax[idx[k]].set(adjustable='box', aspect='equal')
        ax[idx[k]].set_xticks(np.linspace(config.preprocess.dim[0],
                                          config.preprocess.dim[1],
                                          5, endpoint=True))
        ax[idx[k]].set_yticks(np.linspace(config.preprocess.dim[2],
                                          config.preprocess.dim[3],
                                          5, endpoint=True))

    for i in range(nrows):
        for j in range(ncols):
            plt.setp(ax[i, j].get_yticklabels(), visible=False) if j != 0 else \
                None
            plt.setp(ax[i, j].get_xticklabels(), visible=False) if i != 1 else \
                None

    plt.subplots_adjust(hspace=0.1, wspace=0.3)
    plt.savefig('{}/vit_preprocess.png'.format(config.output_dir),
                bbox_inches="tight", dpi=300)
    plt.close()
