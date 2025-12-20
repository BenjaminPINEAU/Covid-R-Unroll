# Unfolded Proximal Algorithms for Estimating the COVID-19 Reproduction Number

This project contains the Python code associated to the report [[rapport_BenjaminPINEAU]](rapport_BenjaminPINEAU.pdf) done during an internship.


## Project description

This project focuses on the **unfolding of the Chambolle–Pock algorithm** to estimate the COVID-19 reproduction number. Several network architectures are available and a toy data set is provided to test the training of the unrolled Chambolle-Pock algorithm. For this purpose, 2 notebooks are also provided : 

- [`train_demo`](train_demo.ipynb)
> Illustrates the general operation of training a network. We also detail the various architectures available.

- [`display_data`](display_data.ipynb)
> Plot the estimators obtained after training the network and comparison with the desired ground truth.

---

## Repository Structure

The repository is organized as follows:

```text
├── train_demo                      # Training scripts and demo experiments
├── display_data                    # Visualization and plotting utilities
└── utils                           
    ├── CP_unfolded
    │   ├── CP_unfolded_Lexp        # Custom neural network
    │   ├── CP_unfolded_LSTexp      # Custom neural network
    │   └── CP_unfolded_OpLin       # Custom neural network
    ├── create_database             # Dataset creation from a given folder path
    ├── config                      # Class for defining settings
    ├── data_gradient_descent       # Gradient descent–based optimization tools
    ├── function_dual_primal        # Primal–dual related functions
    ├── load_data                   # Load specific data from dataset
    ├── oplin                       # Custom linear operators
    └── sliding_median              # Sliding median functions
