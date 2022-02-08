# Continual learning with sketched Jacobian approximations

This repository contains the code for reproducing figures and results in the paper ``Provable Continual Learning via Sketched Jacobian
Approximations''.

# Requirements
The following Python libraries are required to run the code in this repository:

```
numpy
jupyter
torch
torchvision
scipy
```
and can be installed with `pip install -r requirements.txt`.

# Usage

All figures in the paper can be reproduced by running the respective notebooks as indicated below:

**Figure 1**: Sequential learning on the MNIST permutation problem for a neural network and for the random feature model can be reproduced by running the notebooks `continual_learning_mnist_permutation_NN` and `continual_learning_mnist_permutation_random_features`.

**Figure 2**: Sequential learning to classify pairs of MNIST digits can be reproduced by running the `continual_learning_mnist_incremental_random_features` notebook.
    
**Theorem 4**: The risk for the worst case construction is computed in the notebook `continual_learning_toy_example`.



## Citation
```
@inproceedings{,
    author    = {Reinhard Heckel},
    title     = {Provable Continual Learning via Sketched Jacobian Approximations},
    booktitle   = {International Conference on Artificial Intelligence and Statistics (AISTATS)},
    year      = {2022}
}
```

## Licence

All files are provided under the terms of the Apache License, Version 2.0.
