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

from utils.fonction_dual_primal import prox_DKL, proxL1_norm
from utils.oplin import discrete_derivation, discrete_derivation_adj


class CPunfolded_LSTexp(nn.Module):
    
    """
    Define a neural network for the unfolded Chambolle-Pock algorithm 
    parameters to learn are lambda (coeff of l1 regularization) 
    and signa, tau (algorithmic parameter), all different for each layer
    """

    def __init__(self, device, K_layer, choix_optimizer, scheduler = False, lmb_init = 10, learning_rate = 1e-3):
        
        """
        Initialize the network
        INPUT:
        - device: cpu or gpu
        - K_layer: number of layers of the network
        - lmb_init: initial regularisation factor 
        - choix_optimizer: optimizer for the gradient descent 
        - scheduler: adaptative learning rate strategy
        - learning_rate: initial learning rate of the network
        """

        super(CPunfolded_LSTexp, self).__init__()
        self.device = device
        self.K_layer = K_layer
        self.learning_rate = learning_rate

        # Initialize parameters as trainable
        # note that we will learn lambda = exp(ell) (same for sigma and tau) to avoid negative value
        self.param_ell = torch.nn.Parameter(torch.log(torch.tensor(lmb_init))*torch.ones(K_layer, device=self.device)) #lambda
        self.param_u = torch.nn.Parameter(-2*torch.ones(K_layer, device=self.device)) #sigma
        self.param_s = torch.nn.Parameter(-2*torch.ones(K_layer, device=self.device)) #tau

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
            Qk = Qk + torch.exp(self.param_u[k]) * discrete_derivation(Rk_bar) - torch.exp(self.param_u[k])*proxL1_norm(Qk/torch.exp(self.param_u[k]) + discrete_derivation(Rk_bar), torch.exp(self.param_ell[k])/torch.exp(self.param_u[k]))
            Rk1 = self.activation_prox_DKL(Z_norm, PhiZ_norm, Rk - torch.exp(self.param_s[k])*discrete_derivation_adj(Qk, self.device), torch.exp(self.param_s[k])).to(self.device)
            Rk_bar = 2*Rk1 - Rk
            Rk = Rk1

        # Final output after K_layer iterations
        out = Rk

        return out