import numpy as np


def decoder_mask(*, num_patches):
    mask = np.zeros((num_patches * num_patches, num_patches * num_patches))

    k = 0
    l = 0
    for i in range(num_patches):
        for j in np.arange(k, k + (num_patches - 1) * num_patches + num_patches,
                           num_patches):
            mask[l:, j] = 1
            l += 1
        k += 1

    return mask
