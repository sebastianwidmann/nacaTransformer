from matplotlib import pyplot as plt


def visualise(data, field, filename):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    field_idx = {'T': '2', 'alphaT': '3', 'k': '4', 'nut': '5', 'omega': '6',
                 'p': '6',
                 'rho': '7', 'Ux': '9', 'Uy': '9'}

    ax[0].tripcolor(data[:, 0], data[:, 1], data[:, int(field_idx[field])])
    ax[1].scatter(data[:, 0], data[:, 1], s=0.1)

    plt.savefig('../{}.png'.format(filename), dpi=300)
