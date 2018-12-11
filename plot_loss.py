import numpy as np
import random

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages

import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times New Roman']})

# green magenta gold blue red
palette = ["#5ba965", "#c15ca5", "#af963e", "#727cce", "#cb5b4c"]
fnames = ["convnet_alexnet_test_log.txt", "convnet_bucket1_test_log.txt", "convnet_deep_1_test_log.txt",
    "convnet_deep_2_test_log.txt",
    "hybridnet_test_log.txt"]
titles = ["AlexNet", "ConvBucket", "ConvDeep1", "ConvDeep2", "HybridNet"]

for fn, ttl, col in zip(fnames, titles, palette):
    print(f'Reading file {fn}...')
    with open(fn, 'r') as f:
        lines = f.readlines()

    train_lines = list(filter(lambda l: ('Train' in l) and ('images' in l), lines))
    train_lines = list(map(lambda v: float(v.split()[-1]), train_lines))
    test_lines = list(filter(lambda l: ('Test' in l) and ('images' in l), lines))
    test_lines = list(map(lambda v: float(v.split()[-1]), test_lines))

    x = 5 * (np.array(list(range(len(train_lines)))) + 1)
    y = np.array(train_lines)
    z = np.array(test_lines)
    # Hide the right and top spines
    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.plot(x, z, color=col, linewidth=3, label=f'{ttl}')
    # print(f"Plotting plot to {ttl}_train_loss.pdf...")
    # plt.savefig(f'{ttl}_train_loss.png', bbox_inches='tight')
    # plt.clf()

    # ax = plt.subplot(111)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    # # Only show ticks on the left and bottom spines
    # ax.yaxis.set_ticks_position('left')
    # ax.xaxis.set_ticks_position('bottom')
    # plt.plot(x, z, color=col, linewidth=3)
    # plt.xlabel(f'Epoch Number')
    # plt.ylabel(f'MSE Test Loss')
    # plt.title(f'Test Loss vs. Epochs for {ttl}')
    # print(f"Plotting plot {ttl}_test_loss.pdf...")
    # plt.savefig(f'{ttl}_test_loss.png', bbox_inches='tight')
    # plt.clf()

plt.xlabel(f'Epoch Number')
plt.ylabel(f'MSE Test Loss')
plt.title(f'Test Loss vs. Epochs')
plt.legend(loc='lower left')
plt.savefig(f'good_test_loss.png', bbox_inches='tight')
# for fn, ttl in zip(fnames, titles):
#     print(f'Reading file {fn}...')
#     with open(fn, 'r') as f:
#         lines = f.readlines()

#     train_lines = list(filter(lambda l: ('Train' in l) and ('images' in l), lines))
#     train_lines = list(map(lambda v: float(v.split()[-1]), train_lines))
#     test_lines = list(filter(lambda l: ('Test' in l) and ('images' in l), lines))
#     test_lines = list(map(lambda v: float(v.split()[-1]), test_lines))

#     col = random.choice(palette)
#     x = 5 * (np.array(list(range(len(train_lines)))) + 1)
#     y = np.array(train_lines)
#     z = np.array(test_lines)
#     # Hide the right and top spines
#     ax = plt.subplot(111)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)

#     # Only show ticks on the left and bottom spines
#     ax.yaxis.set_ticks_position('left')
#     ax.xaxis.set_ticks_position('bottom')

#     plt.plot(x, y, color=col, linewidth=3)
#     plt.xlabel(f'Epoch Number')
#     plt.ylabel(f'MSE Train Loss')
#     plt.title(f'Train Loss vs. Epochs for {ttl}')
#     print(f"Plotting plot to {ttl}_train_loss.pdf...")
#     plt.savefig(f'{ttl}_train_loss.png', bbox_inches='tight')
#     plt.clf()

#     ax = plt.subplot(111)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)

#     # Only show ticks on the left and bottom spines
#     ax.yaxis.set_ticks_position('left')
#     ax.xaxis.set_ticks_position('bottom')
#     plt.plot(x, z, color=col, linewidth=3)
#     plt.xlabel(f'Epoch Number')
#     plt.ylabel(f'MSE Test Loss')
#     plt.title(f'Test Loss vs. Epochs for {ttl}')
#     print(f"Plotting plot {ttl}_test_loss.pdf...")
#     plt.savefig(f'{ttl}_test_loss.png', bbox_inches='tight')
#     plt.clf()
