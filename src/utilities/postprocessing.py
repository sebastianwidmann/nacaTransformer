from utilities.visualisation import loss_comparison
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
#
# # ----- Weight decay -----
# files = ['output/52/loss_raw.txt', 'output/46/loss_raw.txt',
#          'output/54/loss_raw.txt', 'output/55/loss_raw.txt']
# labels = ['0.0', '0.1', '0.2', '0.3', '0.1 mod.']
#
# loss_comparison(files, labels, 'Weight Decay', 'weight_decay')
#
# # ----- Dropout rate -----
# files = ['output/52/loss_raw.txt', 'output/56/loss_raw.txt',
#          'output/57/loss_raw.txt', 'output/58/loss_raw.txt',
#          'output/65/loss_raw.txt']
#
# loss_comparison(files, labels, 'Dropout', 'dropout')
#
# # ----- Attention dropout rate -----
# files = ['output/52/loss_raw.txt', 'output/59/loss_raw.txt',
#          'output/60/loss_raw.txt', 'output/61/loss_raw.txt']
#
# loss_comparison(files, labels, 'Attention Dropout', 'att_dropout')
#
# # ----- Learning Rate schedulers -----
# files = ['output/52/loss_raw.txt', 'output/62/loss_raw.txt',
#          'output/64/loss_raw.txt', 'output/76/loss_raw.txt']
# labels = ['SGDR 1e-5', 'Cos Decay 1e-5', 'SGDR 1e-6', 'Cos Decay 1e-6']
#
# loss_comparison(files, labels, 'LR Scheduler', 'lr_scheduler')
#
# # ----- Attention heads -----
# files = ['output/66/loss_raw.txt', 'output/67/loss_raw.txt',
#          'output/52/loss_raw.txt', 'output/68/loss_raw.txt',
#          'output/69/loss_raw.txt']
# labels = ['1', '2', '3', '4', '6']
# #
# loss_comparison(files, labels, 'Attention Heads', 'att_heads')
#
# # ----- Layers -----
# files = ['output/70/loss_raw.txt', 'output/71/loss_raw.txt',
#          'output/52/loss_raw.txt', 'output/72/loss_raw.txt',
#          'output/74/loss_raw.txt']
# labels = ['1', '2', '3', '4', '6']
#
# loss_comparison(files, labels, 'Layers', 'layers')
#
# # ----- Patch size} -----
# files = ['output/41/loss_raw.txt', 'output/43/loss_raw.txt']
# labels = ['(5,5)', '(10,10)']
#
# loss_comparison(files, labels, 'Patch Size', 'patch_size')

# # ----- Masked vs Unmasked -----
# files = ['output/38/loss_raw.txt', 'output/40/loss_raw.txt',
#          'output/75/loss_raw.txt']
# labels = ['Transformer', 'Unmasked', 'Flow Direction aligned']
#
# loss_comparison(files, labels, 'Decoder Masking', 'masking')

# # ----- Batch Size -----
# files = ['output/47/loss_raw.txt', 'output/48/loss_raw.txt',
#          'output/49/loss_raw.txt', 'output/50/loss_raw.txt',
#          'output/51/loss_raw.txt']
# labels = ['5', '10', '20', '40', '80']
# #
# loss_comparison(files, labels, 'Batch Size', 'batch_size')

# # ----- Loss Function -----
# files = ['output/39/loss_raw.txt', 'output/45/loss_raw.txt',
#          'output/44/loss_raw.txt']
# labels = ['$L_1$', '$L_2$', 'Huber']
# #
# loss_comparison(files, labels, 'Loss Function', 'lossfn')
#
# # ----- Final configuration -----
# files = ['output/83/loss_raw.txt']
# labels = ['final']
# #
# loss_comparison(files, labels, '', 'final')
# # -------------------------------------

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(
    '/home/sebastianwidmann/Documents/git/nacaTransformer/output/83/loss_mae'
    '.txt',
    delimiter=',')

num_epochs = 200
out_frequency = 25
num_channels = 3

channel_names = ['p', 'ux', 'uy']

num_samples = int(num_epochs / out_frequency)

data = data.reshape((num_samples, -1, 3))

print(data[7, :, 0].mean(), data[7, :, 1].mean(), data[7, :, 2].mean())
print(data[7, :, 0].var(), data[7, :, 1].var(), data[7, :, 2].var())

for i in range(num_channels):
    data_lst = []
    for j in range(num_samples):
        data_lst.append(data[j, :, i])

    data_labels = np.arange(out_frequency, num_epochs + out_frequency,
                            out_frequency)

    fig, ax = plt.subplots(figsize=(10, 5))

    bp = ax.boxplot(data_lst, whis=[1, 99], showfliers=False)
    ax.set_yscale('log')
    ax.set_xticklabels(data_labels)
    ax.set_xlabel('Epoch Number')
    ax.set_ylabel('Test Split Averaged Mean Absolute Error')
    ax.set_ylim(2e-4, 1e-2)

    plt.grid(visible=True, which='major', axis='y', color='#444',
             linestyle='-')
    plt.grid(visible=True, which='minor', axis='y', color='#ccc',
             linestyle='--')

    plt.savefig('vit_boxplot_{}.png'.format(channel_names[i]),
                bbox_inches="tight", dpi=300)
#
# data2 = np.loadtxt(
#     '/home/sebastianwidmann/Documents/git/nacaTransformer/output/83/loss_mse'
#     '.txt',
#     delimiter=',')
#
# data2 = data2.reshape((num_samples, -1, 3))
#
# print(data2[7, :, 0].mean(), data2[7, :, 1].mean(), data2[7, :, 2].mean())
# print(data2[7, :, 0].var(), data2[7, :, 1].var(), data2[7, :, 2].var())
# #
# # print(data[1, :, 1].mean(), data[1, :, 2].mean())
#
# #
# # # fig, ax = plt.subplots(2, 4, figsize=(10, 5))
# #
# # fig, ax = plt.subplots()
# #
# # data_labels = np.arange(25, 225, 25)
# #
# # data = [data[0, :, 1], data[1, :, 1], data[2, :, 1], data[3, :, 1],
# #         data[4, :, 1], data[5, :, 1], data[6, :, 1], data[7, :, 1]]
# #
# # bp = ax.boxplot(data, whis=[1, 99], showfliers=False)
# # ax.set_yscale('log')
# # ax.set_xticklabels(data_labels, rotation=45)
# # ax.set_xlabel('Epoch Number')
#
# # ax = ax.ravel()
# #
# # for idx, a in enumerate(ax):
# #     a.hist(data[idx, :, 0], bins=50)
# #     a.set_xlim(0, 0.03)
# #
# # plt.tight_layout()
# plt.savefig('out.png', dpi=300)
# #
# # print(data[0, :, 2].shape)
# #
# # data25 = data[0, :, 1]
# #
# # print(data[0, :, 1].mean(), data[0, :, 2].mean())
# #
# # q0, q1 = np.percentile(data[0, :, 1], [5, 95])
# # iqr = q1 - q0
# # print(iqr)
# #
# # q0, q1 = np.percentile(data[0, :, 2], [5, 95])
# # iqr = q1 - q0
# # print(iqr)
# # #
# # fig = plt.figure()
# #
# # plt.hist(data[0, :, 2], bins=100)
# # plt.savefig('out.png', dpi=300)
#
# # print(data25.mean(axis=0))
#
# # for i in range(data.shape[0]):
# #     temp = data[i, :, ]
# #
# #     data_lst = [temp[:, 0], temp[:, 1], temp[:, 2]]
# #
# #     fig = plt.figure()
# #
# #     ax = fig.add_subplot(111)
# #
# #     bp = ax.boxplot(data_lst)
# #
# #     ax.set_ylim(0, .05)
# #
# #     plt.savefig('out_{}.png'.format(i), dpi=300)
# #
# #     plt.close()
