# SparseFool
This repository contains the official PyTorch implementation of SparseFool algorithm described in [[1]](https://arxiv.org/abs/1811.02248).

## Requirements

To execute the code, please make sure that the following packages are installed:

- [NumPy](https://docs.scipy.org/doc/numpy-1.15.1/user/install.html)
- [PyTorch and Torchvision](https://pytorch.org/) (install with CUDA if available)
- [matplotlib](https://matplotlib.org/users/installing.html)

## Executing the demos

### test_sparsefool.py

A simple demo that computes the sparse adversarial perturbation of a test image.

### test_perceptibility.py

A simple demo that controls the perceptibility of the resulted perturbation of a test image.

## Contents

### sparsefool.py

This function implements the algorithm proposed in [[1]](https://arxiv.org/abs/1811.02248) using PyTorch to find sparse adversarial perturbations.

The parameters of the function are:

- `im`: image (tensor) of size `1xCxHxW`, where `C` are the channels.
- `net`: neural network.
- `lb`: the lower bounds for the adversarial image values.
- `ub`: the upper bounds for the adversarial image values.
- `lambda_ `: the control parameter for going further into the classification region, by default = 3.
- `max_iter`: max number of iterations, by default = 50.

### linear_solver.py

This function implements the algorithm proposed in [[1]](https://arxiv.org/abs/1811.02248) for solving the linearized box-constrained problem. It is used by sparsefool.py for solving the linearized problem.

### deepfool.py

This function implements the algorithm proposed in [[2]](https://arxiv.org/pdf/1511.04599.pdf) for computing adversarial perturbations. It is used by sparsefool.py for the linear approximation of the decision boundary.

### utils.py

Includes general functions

### data/

Contains some examples for the demos. The images where cropped to have square dimensions:

- `cat.jpg`([source](https://www.hd-wallpapersdownload.com/desktop-hd-cat-and-kittens-pics/)): it is used by test_sparsefool.py.
- `red_light.jpg`([source](https://www.gettyimages.ch/detail/nachrichtenfoto/traffic-light-controls-the-flow-of-vehicles-and-nachrichtenfoto/52663127)): it is used by test_perceptibility.py.

## Reference
[1] A. Modas, S. Moosavi-Dezfooli, P. Frossard:
*SparseFool: a few pixels make a big difference*. In Computer Vision and Pattern Recognition (CVPR ’19), IEEE, 2019.

[2] S. Moosavi-Dezfooli, A. Fawzi, P. Frossard:
*DeepFool: a simple and accurate method to fool deep neural networks*.  In Computer Vision and Pattern Recognition (CVPR ’16), IEEE, 2016.
