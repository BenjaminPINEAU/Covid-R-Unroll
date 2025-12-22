# Unfolded Proximal Algorithm for Estimating the COVID-19 Reproduction Number

This project contains the Python code done during an internship at LS2N (April to August 2025) as part of my studies at Centrale Nantes. The internship was supervised by Barbara Pascal and Sébastien Bourguignon and funded by the CNRS.
This internship led to the writing of a detailed scientific report, *in French*, whose [PDF](https://github.com/BenjaminPINEAU/unrolling-internship/blob/main/docs/rapport_BenjaminPINEAU.pdf) version is available in this repository for completeness.


## Project description

This project focuses on the **unfolding of the Chambolle–Pock algorithm** to estimate the COVID-19 reproduction number. Several network architectures are available and a toy data set is provided to test the training of the unrolled Chambolle-Pock algorithm. For this purpose, 2 notebooks are also provided : 

- [`demo_unroll_train`](demo_unroll_train.ipynb)
> Illustrates the general operation of training a network. We also detail the various architectures available.

- [`demo_R_estim`](demo_R_estim.ipynb)
> Plot the estimators obtained after training the network and comparison with the desired ground truth.

---

## Repository Structure

The repository is organized as follows:

```text
├── demo_unroll_train               # Training scripts and demo experiments
├── demo_R_estim                    # Visualization and plotting utilities
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
