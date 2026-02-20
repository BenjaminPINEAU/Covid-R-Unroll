# Unfolded Proximal Algorithm for Estimating the COVID-19 Reproduction Number

This project contains the Python code associated to the paper
> **Pineau B**, **Pascal, B**, **Bourguignon, S** (2026), "*Unfolded primal-dual algorithm estimating time-varying COVID-19 reproduction numbers*" [`[pdf]`](paper/2026_UnCP_estim.pdf)

[`hal-05512190`](https://hal.science/hal-05512190)

---
## Project description

This project focuses on the **unfolding of the Chambolle–Pock algorithm** to estimate the COVID-19 reproduction number. Four notebooks are also provided: 

- [`demo_compared_architecture`](notebooks/demo_compared_architecture.ipynb)
> Generates a synthetic dataset for a fixed variance level and trains our unfolded neural network with different architectures. This notebook reproduces the experiments leading to Figure n.4 of the article. 

- [`demo_compared_estimators`](notebooks/demo_compared_estimators.ipynb)
> Evaluates the qualitative performance of our unfolded neural network on real COVID-19 counts, with comparison to literature estimators. This notebook reproduces the experiments leading to Figure n.6 of the article.

- [`demo_compared_perf`](notebooks/demo_compared_perf.ipynb)
> Generates synthetic datasets for various variance level, trains our unfolded neural network for a fixed architecture and analyses the quantitative performances with a synthetic test dataset, with comparison to literature estimators. This notebook reproduces the experiments leading to  Figure n.5 of the article.

- [`demo_UnCP_epidemic_monitoring`](notebooks/demo_UnCP_epidemic_monitoring.ipynb)
> Computes the estimator provided by our model for the country and the period of your choice. 

---

## Repository Structure

The repository is organized as follows:

```text
├── data
│    ├── APURE_estimates                 # Folder with computed APURE estimator
│    ├── datasets                        # Folder with synthetic datasets used for the article
│    │   ├── test                        # Test synthetic dataset
│    │   ├── train                       # Train synthetic dataset
│    │   ├── test.csv                    # CSV with dates to recreate synthetic data for the test dataset
│    │   └── train.csv                   # CSV with dates to recreate synthetic data for the train dataset
│    └── models                          # Folder with all models used in the article
├── notebooks
│    ├── demo_compared_architecture      # Notebook to compare different architectures for our proposed estimator
│    ├── demo_compared_estimators         # Notebook to illustrate the qualitative performance of our estimator
│    ├── demo_compared_perf              # Notebook to illustrate the quantitative performance of our estimator
│    └── demo_UnCP_epidemic_monitoring   # Notebook to test our model on the country and period of your choice
└── utils
    ├── config                           # Class for defining settings
    ├── cp_unfolded                      # Custom neural network
    ├── create_database                  # Dataset creation from a given folder path
    ├── create_synthetic_data            # Synthetic data creation functions
    ├── database_gradient_descent        # Gradient descent–based optimization tools
    ├── display_data                     # Function for graph aesthetics
    ├── function_dual_primal             # Primal–dual related functions
    ├── load_data                        # Load specific data from dataset
    ├── metric                           # Function for evaluating the performance of estimators
    ├── oplin                            # Custom linear operators
    ├── preamble                         # Importing the necessary modules for notebooks
    └── sliding_median                   # Sliding median functions