import torchvision.transforms as transforms
import torchvision.models as torch_models
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from utils import nnz_pixels, tf, tf_r, get_label
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
im_orig = Image.open("./data/cat.jpg")
im_sz = 224
im_orig = transforms.Compose([transforms.Resize((im_sz, im_sz))])(im_orig)

# Bounds for Validity and Perceptibility
delta = 255
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

#####################
# Visualize results #
#####################
labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')
str_label_pred = get_label(labels[np.int(pred_label)].split(',')[0])
str_label_fool = get_label(labels[np.int(fool_label)].split(',')[0])

fig, axes = plt.subplots(1, 3)

axes[0].set_title(str_label_pred)
axes[1].set_title("%s pixels" % repr(nnz_pixels(r.cpu().numpy().squeeze())))
axes[2].set_title(str_label_fool)

axes[0].imshow(im_orig)
axes[1].imshow(tf_r(r.cpu().numpy().squeeze(), mean, std))
axes[2].imshow(tf(x_adv.cpu().numpy().squeeze(), mean, std))

axes[0].axis('off')
axes[1].axis('off')
axes[2].axis('off')

plt.show()
plt.close(fig)
