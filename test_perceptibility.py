import torchvision.transforms as transforms
import torchvision.models as torch_models
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from utils import nnz_pixels, inv_tf, get_label
from sparsefool import sparsefool
from utils import valid_bounds
from PIL import Image


# Fix random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.manual_seed(999)
np.random.seed(999)

# Check for cuda devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load a pretrained model
net = torch_models.vgg16(pretrained=True)
net = net.to(device)
net.eval()

# Load Image and Resize
im_orig = Image.open("./data/red_light.jpg")
im_sz = 224
im_orig = transforms.Compose([transforms.Resize((im_sz, im_sz))])(im_orig)
labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

# Bounds for Validity and Perceptibility
delta_l = [255, 50, 25]

fig, axes = plt.subplots(1, 4)
for i in range(len(delta_l)):

    delta = delta_l[i]
    lb, ub = valid_bounds(im_orig, delta)

    # Transform data
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    im = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])(im_orig)
    lb = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])(lb)
    ub = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])(ub)

    im = im[None, :, :, :].to(device)
    lb = lb[None, :, :, :].to(device)
    ub = ub[None, :, :, :].to(device)

    ######################
    # Execute SparseFool #
    ######################
    # Params
    max_iter = 50
    lambda_ = 3.

    x_adv, r, pred_label, fool_label, loops = sparsefool(im, net, lb, ub, lambda_, max_iter)

    # Visualize results
    str_label_pred = get_label(labels[np.int(pred_label)].split(',')[0])
    str_label_fool = get_label(labels[np.int(fool_label)].split(',')[0])

    axes[i + 1].imshow(inv_tf(x_adv.cpu().numpy().squeeze(), mean, std))
    axes[i + 1].set_title("$\delta$: %s" % repr(delta_l[i]))
    axes[i + 1].set_xlabel("%s (%1.2f%% pxls)" % (str_label_fool, 100. * nnz_pixels(r.cpu().numpy().squeeze()) / (im_sz * im_sz)))
    axes[i + 1].xaxis.set_ticks_position('none')
    axes[i + 1].yaxis.set_ticks_position('none')
    axes[i + 1].set_xticklabels([])
    axes[i + 1].set_yticklabels([])

axes[0].imshow(im_orig)
axes[0].set_title("Original")
axes[0].set_xlabel(str_label_pred)
axes[0].xaxis.set_ticks_position('none')
axes[0].yaxis.set_ticks_position('none')
axes[0].set_xticklabels([])
axes[0].set_yticklabels([])

plt.show()
plt.close(fig)
