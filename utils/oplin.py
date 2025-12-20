import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import matplotlib.dates as mdates
from datetime import datetime
import torch.optim as optim


def create_D2(T, device = 'cpu'):
    """
    Create the discrete second derivative linear operator matrix 
    for size T
    """
    D2 = torch.zeros(T-2, T, device = device)
    for i in range(T-2):
        D2[i][i] = 1
        D2[i][i+1] = -2
        D2[i][i+2] = 1
    return D2

def discrete_derivation(R, device = 'cpu'):
    """
    Compute the 2nd discrete derivation of R
    """
    R_zeros = torch.zeros(R.shape, dtype=R.dtype, device = device)
    R_zeros[:, :, :-2] = R[:, :, : -2] - 2*R[:, :, 1: -1] + R[:, :, 2:]
    return R_zeros*0.25
# discrete_derivation = torch.compile(discrete_derivation)

def discrete_derivation_adj(Q, device = 'cpu'):
    """
    Compute the adjoint of the 2nd discrete derivation of R
    """
    Q_zeros = torch.zeros(Q.shape, dtype=Q.dtype, device = device)
    Q_zeros[:, :, 0] = Q[:, :, 0]
    Q_zeros[:, :, 1] = -2*Q[:, :, 0] + Q[:, :, 1]
    Q_zeros[:, :, 2: -2] = Q[:, :, 2: -2] + -2*Q[:, :, 1:-3] + Q[:, :, 0:-4]
    Q_zeros[:, :, -2] = -2*Q[:, :, -3] + Q[:, :, -4]
    Q_zeros[:, :, -1] = Q[:, :, -3]
    
    return Q_zeros*0.25


def op_lin(R, A, B, device = 'cpu'):
    """
    Compute the matrix product MR while M = tridiag(a, b, a)
    """
    R_zeros = torch.zeros(R.shape, dtype=R.dtype, device = device)
    R_zeros[:, :, :-2] = A * R[:, :, : -2] + B * R[:, :, 1: -1] + A * R[:, :, 2:]
    return R_zeros*0.25

def op_lin_adj(Q, A, B, device = 'cpu'):
    """
    Compute the matrix product M*R while M = tridiag(a, b, a)
    """
    Q_zeros = torch.zeros(Q.shape, dtype=Q.dtype, device = device)
    Q_zeros[:, :, 0] = A*Q[:, :, 0]
    Q_zeros[:, :, 1] = B*Q[:, :, 0] + A*Q[:, :, 1]
    Q_zeros[:, :, 2: -2] = A * Q[:, :, 2: -2] + B * Q[:, :, 1:-3] + A * Q[:, :, 0:-4]
    Q_zeros[:, :, -2] = A*Q[:, :, -3] + B*Q[:, :, -4]
    Q_zeros[:, :, -1] = A*Q[:, :, -3]
    
    return Q_zeros*0.25

def op_lin_mat(R, A, device = 'cpu'):
    """
    Compute the matrix product AR
    """
    T = R.shape[-1]
    R_zeros = torch.zeros(R.shape, dtype=R.dtype, device = device)
    R_zeros[:, :, :-2] = torch.einsum('ot,bst->bso', A, R)
    return R_zeros*0.25


def op_lin_adj_mat(Q, A):
    """
    Compute the matrix product AQ
    """
    return torch.einsum('ot,bst->bso', A, Q[:, :, :-2])*0.25


def op_lin_conv(R, vect_op_lin):
    """
    Compute the convolution h*R while h = (c_1, ..., c_25)
    """
    y = torch.nn.functional.conv1d(R, vect_op_lin)
    return y

def op_lin_adj_conv(Q, vect_op_lin):
    """
    Compute the transposed convolution h*R while h = (c_1, ..., c_25)
    """
    y = torch.nn.functional.conv_transpose1d(Q, vect_op_lin)
    return y


def op_lin_convS(R, vect_op_lin_start):
    return torch.einsum('ot,bst->bso', vect_op_lin_start, R[:, :, :25])

def op_lin_convE(R, vect_op_lin_end):
    return torch.einsum('ot,bst->bso', vect_op_lin_end, R[:, :, -25:])

def op_lin_adj_convS(Q_start, vect_op_lin_start):
    return torch.einsum('ot,bst->bso', torch.transpose(vect_op_lin_start, 0, 1), Q_start)

def op_lin_adj_convE(Q_end, vect_op_lin_end):
    return torch.einsum('ot,bst->bso', torch.transpose(vect_op_lin_end, 0, 1), Q_end)