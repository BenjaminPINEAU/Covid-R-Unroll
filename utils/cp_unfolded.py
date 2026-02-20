import torch
import torch.nn as nn

from function_dual_primal import prox_DKL, proxL1_norm
from oplin import op_lin_conv, op_lin_adj_conv


class CPunfolded_conv(nn.Module):

    """
    Define a neural network for the unfolded Chambolle-Pock algorithm 
    Parameters to be learned are filters, all different for each layer
    """

    def __init__(self, device, K_layer, learning_rate = 1e-4, nb_filter = 20):
        """
        Initialize the network
        INPUT:
        - device: cpu or gpu
        - K_layer: number of layers of the network
        - optimizer_choice: optimizer for the gradient descent 
        - scheduler: adaptative learning rate strategy
        - learning_rate: initial learning rate of the network
        - nb_filter: number of filter applied in parallel
        """
                
        super(CPunfolded_conv, self).__init__()
        self.device = device
        self.learning_rate = learning_rate
        self.K_layer = K_layer
        self.nb_filter = nb_filter

        # Initialize parameters as trainable
        pattern = torch.tensor([1, 2, 1] + [0]*22, dtype=torch.float, device=self.device)
        pattern = pattern.view(1, 1, 1, 25)
        tensor = pattern.expand(K_layer, nb_filter, 1, 25).clone()
        self.param_op_lin = nn.Parameter((1/(nb_filter))*tensor)

        # Initialize optimizer for the parameters
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
   

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

        Rk = R_initial
        Qk = op_lin_conv(Rk, self.param_op_lin[0])
        Rk_bar = Rk.clone()


        # Forward through each layer
        for k in range(self.K_layer):
            Qk = Qk + 0.99 * op_lin_conv(Rk_bar, self.param_op_lin[k]) - 0.99*proxL1_norm(Qk/0.99 +op_lin_conv(Rk_bar, self.param_op_lin[k]), 1/0.99)
            Rk1 = prox_DKL(Z_norm, PhiZ_norm, Rk - 0.99*op_lin_adj_conv(Qk, self.param_op_lin[k]), 0.99)
            Rk_bar = 2*Rk1 - Rk
            Rk = Rk1
            
        # Final output after K_layer iterations
        out = Rk

        return out