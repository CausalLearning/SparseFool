import torch as torch
import copy
from linear_solver import linear_solver
from torch.autograd import Variable
from utils import clip_image_values
from deepfool import deepfool


def sparsefool(x_0, net, lb, ub, lambda_=3., max_iter=20, epsilon=0.02, device='cuda'):

    pred_label = torch.argmax(net.forward(Variable(x_0, requires_grad=True)).data).item()

    x_i = copy.deepcopy(x_0)
    fool_im = copy.deepcopy(x_i)

    fool_label = pred_label
    loops = 0

    while fool_label == pred_label and loops < max_iter:

        normal, x_adv = deepfool(x_i, net, lambda_, device=device)

        x_i = linear_solver(x_i, normal, x_adv, lb, ub)

        fool_im = x_0 + (1 + epsilon) * (x_i - x_0)
        fool_im = clip_image_values(fool_im, lb, ub)
        fool_label = torch.argmax(net.forward(Variable(fool_im, requires_grad=True)).data).item()

        loops += 1

    r = fool_im - x_0
    return fool_im, r, pred_label, fool_label, loops
