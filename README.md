# SparseFool
SparseFool is a fast and efficient algorithm for sparse adversarial perturbations.

### sparsefool.py

This function implements the algorithm proposed in [[1]]() using PyTorch to find sparse adversarial perturbations.

The parameters of the function are:

- `im`: image (tensor) of size `1xHxWx3d`.
- `net`: neural network.
- `lb`: the lower bounds for the adversarial image values.
- `ub`: the upper bounds for the adversarial image values.
- `lambda_ `: the factor that moves the hyperplane deeper into the classification region, by default = 3.
- `max_iter`: max number of iterations, by default = 50.

### linear_solver.py

This function implements the other algorithm proposed in [[1]]() for solving the linearized box-constrained problem. It is used by sparsefool.py for solving the linearized problem.

### deepfool.py

This function implements the algorithm proposed in [[2]]() for computing adversarial perturbations. It is used by sparsefool.py for the linear approximation of the decision boundary.

### test_sparsefool.py

A simple demo which computes the sparse adversarial perturbation of a test image.

### test_perceptibility.py

A simple demo that controls the perceptibility of the resulted perturbation of a test image.

## Reference
[1]
[2] S. Moosavi-Dezfooli, A. Fawzi, P. Frossard:
*DeepFool: a simple and accurate method to fool deep neural networks*.  In Computer Vision and Pattern Recognition (CVPR ’16), IEEE, 2016.