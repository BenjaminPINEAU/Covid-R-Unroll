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
import time

from utils.fonction_dual_primal import prox_DKL, proxL1_norm, proxL1_norm_adj
from utils.oplin import create_D2, op_lin, op_lin_mat, op_lin_adj, op_lin_adj_mat, op_lin_conv, op_lin_adj_conv


class CPunfolded_OpLin(nn.Module):
    
    """
    Define a neural network for the unfolded Chambolle-Pock algorithm 
    parameters to learn are sigma, tau (algorithmic parameters), 
    and the linear operator M = tridiag(a, b, a), all different for each layer
    """

    def __init__(self, device, K_layer, choix_optimizer, scheduler = False, lmb_init = 10, learning_rate = 1e-3):
        
        """
        Initialize the network
        INPUT:
        - device: cpu or gpu
        - K_layer: number of layers of the network
        - choix_optimizer: optimizer for the gradient descent 
        - scheduler: adaptative learning rate strategy
        - lmb_init: initial regularisation factor 
        - learning_rate: initial learning rate of the network
        """

        super(CPunfolded_OpLin, self).__init__()
        self.device = device
        self.learning_rate = learning_rate
        self.K_layer = K_layer

        # Initialize parameters as trainable
        self.param_ell = torch.nn.Parameter(torch.log(torch.tensor(lmb_init))*torch.ones(K_layer, device = self.device)) #lambda
        self.param_u = torch.nn.Parameter(-2*torch.ones(K_layer, device = self.device)) #sigma
        self.param_s = torch.nn.Parameter(-2*torch.ones(K_layer, device = self.device)) #tau

        self.D2_a = torch.nn.Parameter(torch.ones(K_layer, device = self.device))
        self.D2_b = torch.nn.Parameter(-2*torch.ones(K_layer, device = self.device))

        self.D2adj_a = torch.nn.Parameter(torch.ones(K_layer, device = self.device))
        self.D2adj_b = torch.nn.Parameter(-2*torch.ones(K_layer, device = self.device))

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
        Qk = op_lin(Rk, self.D2_a[0], self.D2_b[0])
        Rk_bar = Rk.clone()

        # Forward through each layer
        for k in range(self.K_layer):
            Qk = Qk + torch.exp(self.param_u[k]) * op_lin(Rk_bar, self.D2_a[k], self.D2_b[k]) - torch.exp(self.param_u[k])*proxL1_norm(Qk/torch.exp(self.param_u[k]) + op_lin(Rk_bar, self.D2_a[k], self.D2_b[k]), 1/torch.exp(self.param_u[k]))
            Rk1 = self.activation_prox_DKL(Z_norm, PhiZ_norm, Rk - torch.exp(self.param_s[k])*op_lin_adj(Qk, self.D2_a[k], self.D2_b[k], self.device), torch.exp(self.param_s[k])).to(self.device)
            Rk_bar = 2*Rk1 - Rk
            Rk = Rk1

        # Final output after K_layer iterations
        out = Rk

        return out
    


class CPunfolded_OpLin_vect(nn.Module):

    """
    Define a neural network for the unfolded Chambolle-Pock algorithm 
    parameters to learn are sigma, tau (algorithmic parameters), 
    and the linear operator as a filter of a size 25, all different for each layer
    """

    def __init__(self, device, K_layer, choix_optimizer, learning_rate = 1e-3, scheduler = False):

        """
        Initialize the network
        INPUT:
        - device: cpu or gpu
        - K_layer: number of layers of the network
        - choix_optimizer: optimizer for the gradient descent 
        - learning_rate: initial learning rate of the network
        - scheduler: adaptative learning rate strategy
        """

        super(CPunfolded_OpLin_vect, self).__init__()
        self.device = device
        self.learning_rate = learning_rate
        self.K_layer = K_layer

        # Initialize parameters as trainable
        self.param_u = torch.nn.Parameter(-2*torch.ones(K_layer, device = self.device)) #sigma
        self.param_s = torch.nn.Parameter(-2*torch.ones(K_layer, device = self.device)) #tau

        self.param_op_lin = nn.Parameter(torch.randn(K_layer, 1, 1, 25))

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
        Qk = op_lin_conv(Rk, self.param_op_lin[0])
        Rk_bar = Rk.clone()

        # Forward through each layer
        for k in range(self.K_layer):
            Qk = Qk + torch.exp(self.param_u[k]) * op_lin_conv(Rk_bar, self.param_op_lin[k]) - torch.exp(self.param_u[k])*proxL1_norm(Qk/torch.exp(self.param_u[k]) + op_lin_conv(Rk_bar, self.param_op_lin[k]), 1/torch.exp(self.param_u[k]))
            Rk1 = self.activation_prox_DKL(Z_norm, PhiZ_norm, Rk - torch.exp(self.param_s[k])*op_lin_adj_conv(Qk, self.param_op_lin[k]), torch.exp(self.param_s[k]))
            Rk_bar = 2*Rk1 - Rk
            Rk = Rk1

        # Final output after K_layer iterations
        out = Rk

        return out
    


class CPunfolded_OpLin_mat(nn.Module):

    """
    Define a neural network for the unfolded Chambolle-Pock algorithm 
    parameter to learn is the entire linear operator
    """

    def __init__(self, device, T, K_layer, choix_optimizer, scheduler = False, learning_rate = 1e-3):
        
        """
        Initialize the network
        INPUT:
        - device: cpu or gpu
        - T: size of the temporal serie
        - K_layer: number of layers of the network
        - choix_optimizer: optimizer for the gradient descent 
        - scheduler: adaptative learning rate strategy
        - learning_rate: initial learning rate of the network
        """

        super(CPunfolded_OpLin_mat, self).__init__()
        self.device = device
        self.T = T
        self.learning_rate = learning_rate
        self.K_layer = K_layer

        # Initialize parameters as trainable
        self.D2 = torch.nn.Parameter(create_D2(self.T, self.device))
        self.D2adj = torch.nn.Parameter(torch.t(create_D2(self.T, self.device)))

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
        Qk = op_lin_mat(Rk, self.D2)
        Rk_bar = Rk.clone()

        # Forward through each layer
        for k in range(self.K_layer):
            Qk = Qk + 0.99 * op_lin_mat(Rk_bar, self.D2) - 0.99*proxL1_norm(Qk/0.99 + op_lin_mat(Rk_bar, self.D2), 1/0.99)
            Rk1 = self.activation_prox_DKL(Z_norm, PhiZ_norm, Rk - 0.99*op_lin_adj_mat(Qk, torch.t(self.D2)), 0.99).to(self.device)
            Rk_bar = 2*Rk1 - Rk
            Rk = Rk1

        # Final output after K_layer iterations
        out = Rk

        return out