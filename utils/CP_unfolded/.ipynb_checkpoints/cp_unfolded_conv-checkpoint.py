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
from utils.metric import loss_L1, loss_L2, second_derivative_loss
from utils.oplin import op_lin_conv, op_lin_convS, op_lin_convE, op_lin_adj_convS, op_lin_adj_convE, op_lin_adj_conv


class CPunfolded_OpLin_convSE(nn.Module):

    """
    Define a neural network for the unfolded Chambolle-Pock algorithm 
    parameter to learn are multi-filter, all different for each layer, to mimic CNN
    """

    def __init__(self, device, K_layer, choix_loss, choix_optimizer, scheduler = False, learning_rate = 1e-3, largeur = 20, start_end_oplin_size = 20):

        """
        Initialize the network
        INPUT:
        - device: cpu or gpu
        - K_layer: number of layers of the network
        - choix_loss: L1 or L2
        - choix_optimizer: optimizer for the gradient descent 
        - scheduler: adaptative learning rate strategy
        - learning_rate: initial learning rate of the network
        """
                
        super(CPunfolded_OpLin_convSE, self).__init__()
        self.device = device
        self.learning_rate = learning_rate
        self.K_layer = K_layer
        self.largeur = largeur

        # Initialize parameters as trainable
        pattern = torch.tensor([1, 2, 1] + [0]*22, dtype=torch.float, device=self.device)
        pattern = pattern.view(1, 1, 1, 25)
        tensor = pattern.expand(K_layer, largeur, 1, 25).clone()
        self.param_op_lin = nn.Parameter((1/largeur)*tensor)

        self.param_op_lin_start = nn.Parameter(torch.zeros(K_layer, start_end_oplin_size, 25, device=self.device))

        self.param_op_lin_end = nn.Parameter(torch.zeros(K_layer, start_end_oplin_size, 25, device=self.device))

        # Initialize activation functions
        self.activation_prox_DKL = prox_DKL
        self.activation_prox_L1normadj = proxL1_norm_adj

        # Initialize optimizer for the parameters
        if choix_optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        if choix_optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr = self.learning_rate)
        if choix_optimizer == "adamW":
            self.optimizer = torch.optim.AdamW(self.parameters(), lr = self.learning_rate)
        if scheduler:
            #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min = 5*10**(-5))
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
            #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma = 0.9)
        else : 
            self.scheduler = False
        self.step_count = 0
        
        # Initialize loss function
        if choix_loss == 'L2':
            self.loss_fn = loss_L2()
        if choix_loss == 'L1':
            self.loss_fn = loss_L1()
    
    def step_loss(self, R_true, out):
        lr = 1e-3
        if self.loss_fn(R_true, out) <= 1e-4:
            lr = 1e-4
        elif self.loss_fn(R_true, out) <= 1e-5:
            lr = 1e-5
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def grow(self, step = 20):
        self.active_neurones = min(self.largeur, self.active_neurones + step)

   
    def forward(self, R_initial, R_true, Z_norm, PhiZ_norm):

        """
        Forward pass of the network
        INPUT:
        - R_initial: initial reproduction number
        - R_true: "groundtruth"
        - Z_norm: normalized daily count (by the standard deviation of the daily counts Z)
        - PhiZ_norm: normalized global infectiousness (by the standard deviation of the daily counts Z)
    
        OUTPUT:
        - out: final estimator of the reproduction number
        - total_loss: loss between out and the groundtruth R_true
        """

        Rk = R_initial

        Qk = op_lin_conv(Rk, self.param_op_lin[0])
        Qk_start = op_lin_convS(Rk, self.param_op_lin_start[0])
        Qk_end = op_lin_convS(Rk, self.param_op_lin_end[0])

        Rk_bar = Rk.clone()

        # Forward through each layer

        for k in range(self.K_layer):

            Qk = Qk + 0.99 * op_lin_conv(Rk_bar, self.param_op_lin[k]) - 0.99*proxL1_norm(Qk/0.99 +op_lin_conv(Rk_bar, self.param_op_lin[k]), 1/0.99)
            Qk_start = Qk_start + 0.99 * op_lin_convS(Rk_bar, self.param_op_lin_start[k]) - 0.99*proxL1_norm(Qk_start/0.99 +op_lin_convS(Rk_bar, self.param_op_lin_start[k]), 1/0.99)
            Qk_end = Qk_end + 0.99 * op_lin_convE(Rk_bar, self.param_op_lin_end[k]) - 0.99*proxL1_norm(Qk_end/0.99 +op_lin_convE(Rk_bar, self.param_op_lin_end[k]), 1/0.99)

            R_start_temp = op_lin_adj_convS(Qk_start, self.param_op_lin_start[0])
            R_end_temp = op_lin_adj_convE(Qk_end, self.param_op_lin_end[0])
            diff_start = Rk.size(-1) - R_start_temp.size(-1)
            R_start_temp = torch.nn.functional.pad(R_start_temp, (0, diff_start))
            diff_end = Rk.size(-1) - R_end_temp.size(-1)
            R_end_temp = torch.nn.functional.pad(R_end_temp, (diff_end, 0))

            Rk1 = self.activation_prox_DKL(Z_norm, PhiZ_norm, Rk - 0.99*(op_lin_adj_conv(Qk, self.param_op_lin[k]) + R_start_temp + R_end_temp), 0.99)
            Rk_bar = 2*Rk1 - Rk
            Rk = Rk1
            
        # Final output after K_layer iterations
        out = Rk

        def trace_grad(grad):
            print(f"Gradient après opérations : {grad[0][0]}")
            return grad

        #self.D2.register_hook(trace_grad)

        return out