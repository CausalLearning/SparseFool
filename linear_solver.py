import numpy as np
import torch
import copy
from utils import clip_image_values


def linear_solver(x_0, normal, boundary_point, lb, ub):

    input_shape = x_0.size()

    coord_vec = copy.deepcopy(normal)
    plane_normal = copy.deepcopy(coord_vec).view(-1)
    plane_point = copy.deepcopy(boundary_point).view(-1)

    x_i = copy.deepcopy(x_0)

    f_k = torch.dot(plane_normal, x_0.view(-1) - plane_point)
    sign_true = f_k.sign().item()

    beta = 0.001 * sign_true
    current_sign = sign_true

    while current_sign == sign_true and coord_vec.nonzero().size()[0] > 0:

        f_k = torch.dot(plane_normal, x_i.view(-1) - plane_point) + beta

        pert = f_k.abs() / coord_vec.abs().max()

        mask = torch.zeros_like(coord_vec)
        mask[np.unravel_index(torch.argmax(coord_vec.abs()), input_shape)] = 1.

        r_i = torch.clamp(pert, min=1e-4) * mask * coord_vec.sign()

        x_i = x_i + r_i
        x_i = clip_image_values(x_i, lb, ub)

        f_k = torch.dot(plane_normal, x_i.view(-1) - plane_point)
        current_sign = f_k.sign().item()

        coord_vec[r_i != 0] = 0

    return x_i
