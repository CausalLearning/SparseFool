import numpy as np
import torch
import copy


def clip_image_values(x, minv, maxv):

    x = torch.max(x, minv)
    x = torch.min(x, maxv)
    return x


def valid_bounds(img, delta=255):

    im = copy.deepcopy(np.asarray(img))
    im = im.astype(np.int)

    # General valid bounds [0, 255]
    valid_lb = np.zeros_like(im)
    valid_ub = np.full_like(im, 255)

    # Compute the bounds
    lb = im - delta
    ub = im + delta

    # Validate that the bounds are in [0, 255]
    lb = np.maximum(valid_lb, np.minimum(lb, im))
    ub = np.minimum(valid_ub, np.maximum(ub, im))

    # Change types to uint8
    lb = lb.astype(np.uint8)
    ub = ub.astype(np.uint8)

    return lb, ub


def inv_tf(x, mean, std):

    for i in range(len(mean)):

        x[i] = np.multiply(x[i], std[i], dtype=np.float32)
        x[i] = np.add(x[i], mean[i], dtype=np.float32)

    x = np.swapaxes(x, 0, 2)
    x = np.swapaxes(x, 0, 1)

    return x


def inv_tf_pert(r):

    pert = np.sum(np.absolute(r), axis=0)
    pert[pert != 0] = 1

    return pert


def get_label(x):
    s = x.split(' ')
    label = ''
    for l in range(1, len(s)):
        label += s[l] + ' '

    return label


def nnz_pixels(arr):
    return np.count_nonzero(np.sum(np.absolute(arr), axis=0))
