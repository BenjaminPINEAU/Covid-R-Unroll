import deepinv as dinv
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
mpl.rcParams.update(mpl.rcParamsDefault)
from PIL import Image
from deepinv.loss.metric import PSNR
perf_psnr = PSNR()
import torch.nn as nn
from torchviz import make_dot
import math

from utils.fonction_dual_primal import prox_DKL, proxL1_norm
from utils.metric import loss_L1, loss_L2
from utils.oplin import discrete_derivation, discrete_derivation_adj


class CPunfolded(nn.Module):
    
    """
    Define a neural network for the unfolded Chambolle-Pock algorithm 
    parameter to learn is lambda (coeff of l1 regularization)
    """

    def __init__(self, device, tau, sigma, K_layer, choix_optimizer, scheduler = False, lmb_init = 1, learning_rate = 1e-3):
        
        """
        Initialize the network
        INPUT:
        - device: cpu or gpu
        - tau, sigma: algorithmic parameters
        - K_layer: number of layers of the network
        - lmb_init: initial regularisation factor 
        - learning_rate: initial learning rate of the network
        - choix_optimizer: optimizer for the gradient descent 
        - scheduler: adaptative learning rate strategy
        """

        super(CPunfolded, self).__init__()
        self.device = device
        self.param_sigma = sigma
        self.param_tau = tau
        self.K_layer = K_layer
        self.learning_rate = learning_rate

        # Initialize parameters as trainable
        self.param_lambda = torch.nn.Parameter(lmb_init*torch.ones(1, device=self.device)) #lambda

        # Initialize activation functions
        self.activation_prox_DKL = prox_DKL

        # Initialize optimizer for the parameters
        if choix_optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        if choix_optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr = self.learning_rate)
        if scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min = 5*10**(-5))
        else : 
            self.scheduler = False

   
    def forward(self, R_initial, Z_norm, PhiZ_norm):

        """
        Forward pass of the network
        INPUT:
        - R_initial: initial reproduction number
        - Z_norm: normalized daily count (by the standard deviation of the daily counts Z)
        - PhiZ_norm: normalized global infectiousness (by the standard deviation of the daily counts Z)
    
        OUTPUT:
        - out: final estimator of the reproduction number
        """

        # Initialize variables
        Rk = R_initial.to(self.device)
        Qk = discrete_derivation(Rk)
        Rk_bar = Rk.clone()

        # Forward through each layer
        for k in range(self.K_layer):
            Qk = Qk + self.param_sigma * discrete_derivation(Rk_bar) - self.param_sigma*proxL1_norm(Qk/self.param_sigma + discrete_derivation(Rk_bar), self.param_lambda/self.param_sigma)
            Rk1 = self.activation_prox_DKL(Z_norm, PhiZ_norm, Rk - self.param_tau*discrete_derivation_adj(Qk, self.device), self.param_tau).to(self.device)
            Rk_bar = 2*Rk1 - Rk
            Rk = Rk1

        # Final output after K_layer iterations
        out = Rk

        return out