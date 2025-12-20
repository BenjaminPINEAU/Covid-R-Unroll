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
from scipy.io import loadmat
from torch.utils.data import DataLoader
import pandas as pd
import copy
import time
import os

from utils.CP_unfolded import cp_unfolded_l as cp_l, cp_unfolded_LSTexp as cp_lstexp, cp_unfolded_Lexp as cp_lexp, cp_unfolded_OpLin as cp_oplin, cp_unfolded_conv as cp_conv
from utils.oplin import discrete_derivation
from utils.metric import metric
from utils.fonction_dual_primal import CP_optim_fix
from utils.create_database import CustomDataset, CustomDataset_window


def load_model(model, init_param):

    """
    Sets the model parameters to certain values

    INPUT:
    - model: the model we use
    - init_param: the value of the parameters to be set

    OUTPUT:
    - the model sets with the specifics values
    """

    param_model = torch.load(init_param, map_location="cpu")
    model.load_state_dict(param_model['model_state_dict'], strict = False)
    model.optimizer.load_state_dict(param_model['optimizer_state_dict'])
    if model.scheduler:
        model.scheduler.load_state_dict(param_model['scheduler_state_dict'])
    

def gradient_descent_database(config):

    """    
    This function operate a gradient descent on the selected network to learn parameters on a database
    and give a prediction of R

    INPUT: 
    - device: cpu or gpu 
    - dataset_train: custom dataset for training
    - dataset_validation: custom dataset for validation
    - nbatch: size of the batch
    - choix_model: type of network used
    - choix_loss: L1 or L2
    - nepoch: number of epoch to learn the parameters
    - K_layer: depth of the network
    - lmb_init: initial value of lambda
    - tau & sigma: convergence parameters og the CP algorithm
    - init_param: file to initialise the parameters of the selected network
    - scheduler: adaptative learning rate strategy
    - learning_rate: initial learning rate of the network
    - optimizer: optimizer for the gradient descent

    OUTPUT:
    - model: final model
    - model_best: best model wrt training loss
    - model_best_validation: best model wrt validation loss
    - out: final prediction of R
    - loss_value_train: loss through the epochs on training dataset
    - loss_value_validation: loss throungh the epochs on validation dataset
    - dKL_value: objectiv function through the epochs
    - pen_value: penalization value through the epochs
    """

    nbatch_train = config.nbatch_train
    nbatch_validation = config.nbatch_validation
    nepoch = config.nepoch
    choix_model = config.choix_model
    choix_loss = config.choix_loss
    device = config.device
    K_layer = config.K_layer
    sigma = config.sigma
    tau = config.tau
    lmb_init = config.lmb_init
    optimizer = config.optimizer
    scheduler = config.scheduler
    learning_rate = config.learning_rate
    largeur = config.largeur
    init_param = config.init_param
    start_end_oplin_size = config.start_end_oplin_size

    dataset_train = CustomDataset(config.dataset_train, device = device, norm_type = config.norm_type)
    dataset_validation = CustomDataset(config.dataset_validation, device = device,  norm_type = config.norm_type)
    #g = torch.Generator()
    #g.manual_seed(seed)
    dataloader_train = DataLoader(dataset_train, batch_size = nbatch_train, shuffle = True, num_workers = 0)#, generator = g)
    dataloader_validation = DataLoader(dataset_validation, batch_size = nbatch_validation, shuffle = True, num_workers = 0)


    # Initialize parameters
    loss_value_train = 1e10 * torch.ones(nepoch) 
    loss_value_validation = 1e10 * torch.ones(nepoch)
    loss_value_train_all = 1e10 * torch.ones(nepoch) 
    loss_value_validation_all = 1e10 * torch.ones(nepoch) 
    dKL_value = 1e10 * torch.ones(nepoch)
    metric_value = 1e10 * torch.ones(nepoch)
    norm_value_list = 1e10 * torch.ones(nepoch)
    norm_value = []

    iteration = 0 
    c_increment = 0
    cmax_increment = 25
    c_loss = 0
    cmax_loss = 100
    epsilon_increment = 1e-5
    epsilon_loss = 1e-10
    loss_old = torch.tensor(1e10)
    loss_train = torch.tensor(1e10)
    loss_validation = torch.tensor(1e10)
    cumul_grad = 4
    R_init_type = config.R_init_type

    # create the folder to save data during the training
    save_dir = "save"
    os.makedirs(save_dir, exist_ok=True)


    for R, Z, PhiZ in dataloader_train:
        T = int(R.shape[-1])
        break

    # Initialize the model
    if choix_model == 'Lexp':
        model = cp_lexp.CPunfolded_Lexp(device, sigma, tau, K_layer, choix_loss, lmb_init)
    elif choix_model == 'LSTexp':
        model = cp_lstexp.CPunfolded_LSTexp(device, K_layer, choix_loss, lmb_init)
    elif choix_model == 'l':
        model = cp_l.CPunfolded(device, sigma, tau, K_layer, choix_loss, lmb_init)
    elif choix_model == 'OpLin':
        model = cp_oplin.CPunfolded_OpLin(device, K_layer, choix_loss, lmb_init)
    elif choix_model == 'OpLin_mat':
        model = cp_oplin.CPunfolded_OpLin_mat(device, T, K_layer, choix_loss, optimizer, scheduler, learning_rate)
    elif choix_model == 'OpLin_vect':
        model = cp_oplin.CPunfolded_OpLin_vect(device, K_layer, choix_loss, lmb_init)
    elif choix_model == 'OpLin_conv':
        model = cp_oplin.CPunfolded_OpLin_conv(device, K_layer, choix_loss, optimizer, learning_rate = learning_rate, scheduler = scheduler, largeur = largeur)
    elif choix_model == 'Conv':
        model = cp_conv.CPunfolded_OpLin_convSE(device, K_layer, choix_loss, optimizer, learning_rate = learning_rate, scheduler = scheduler, largeur = largeur, start_end_oplin_size = start_end_oplin_size)
    elif choix_model == 'OpLin_convTS':
        model = cp_oplin.CPunfolded_OpLin_convTS(device, K_layer, choix_loss, optimizer, learning_rate = learning_rate, scheduler = scheduler, largeur = largeur)

    else : 
        print('Error in choix')
        return False

    # Adjust the model settings if necessary
    if init_param:
        load_model(model, init_param)

    # model.grow(step = 0)
    # print(model.active_neurones)

    # Iteration for each epochs
    model.optimizer.zero_grad()
    
    while iteration < nepoch:
        loss_all = 0
        total_samples = 0
        # Iteration for each data in the batch
        for i, (R_lmb_batch, Z_batch, PhiZ_batch) in enumerate(dataloader_train):
            if R_init_type == 'MLE':
                R_batch = Z_batch/PhiZ_batch
            elif R_init_type == 'CP':
                R_batch = CP_optim_fix(Z_batch, PhiZ_batch, 0.99, 0.99, 50, 500)
            elif R_init_type == 'ones':
                R_batch = torch.ones_like(Z_batch)

            # Zero the gradients
            model.optimizer.zero_grad()

            # Forward pass
            #with torch.autograd.set_detect_anomaly(True):
            out = model.forward(R_batch, R_lmb_batch, Z_batch, PhiZ_batch)
            if choix_loss == 'L2':
                total_loss = torch.nn.functional.mse_loss(out, R_lmb_batch)
            elif choix_loss == 'L1':
                total_loss = torch.nn.functional.l1_loss(out, R_lmb_batch)

            # Backward pass
            total_loss.backward()
            model.optimizer.step()

        model.eval()
        for i, (R_lmb_batch, Z_batch, PhiZ_batch) in enumerate(dataloader_train):
            if R_init_type == 'MLE':
                R_batch = Z_batch/PhiZ_batch
            elif R_init_type == 'CP':
                R_batch = CP_optim_fix(Z_batch, PhiZ_batch, 0.99, 0.99, 50, 500)
            elif R_init_type == 'ones':
                R_batch = torch.ones_like(Z_batch)

            # Forward pass
            #with torch.autograd.set_detect_anomaly(True):
            out = model.forward(R_batch, R_lmb_batch, Z_batch, PhiZ_batch)
            if choix_loss == 'L2':
                total_loss = torch.nn.functional.mse_loss(out, R_lmb_batch)
            elif choix_loss == 'L1':
                total_loss = torch.nn.functional.l1_loss(out, R_lmb_batch)

            current_batch_size = Z_batch.size(-1)
            loss_all = loss_all + total_loss.item()*current_batch_size
            total_samples = total_samples + current_batch_size

        if torch.isnan(out).any():
            print("NaN dans l'output")
            break
        if torch.isnan(total_loss).any():
            print("NaN dans la loss")
            break

        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm ** 2
        total_norm = total_norm ** 0.5
        #print(f"||grad|| = {total_norm:.4f}")

        loss_value_train[iteration] = total_loss.item()
        loss_value_train_all[iteration] = loss_all/total_samples
        norm_value_list[iteration] = total_norm

        # if total_loss.item() < 1e-1:
        #      for param_group in model.optimizer.param_groups:
        #          param_group['lr'] = 1e-4
        
        if scheduler:
            model.scheduler.step()  
        # Stop criteria
        # test for the increment criteria
        if torch.abs(loss_old - total_loss)/torch.abs(loss_old) < epsilon_increment:
            c_increment = c_increment + 1
        else:
            c_increment = 0

        # test for the loss criteria
        if total_loss < epsilon_loss:
            c_loss = c_loss + 1
        else:
            c_loss = 0

        # print the reason of the early stopping
        #if c_increment >= cmax_increment:
        #    print("Reason for stopping : ")
        #    print("Stabilisation of the increments")
        #    break
        if c_loss >= cmax_loss:
            print("Reason for stopping : ")
            print("Loss < e-10")
            break
        
        # Save the best model on the training dataset
        # if total_loss < loss_train:
        #     #norm_value.append(torch.norm(model.D2, p = 'fro'))
        #     best_state_dict = copy.deepcopy(model.state_dict())
        #     loss_train = total_loss

        loss_old = total_loss
        if iteration % 100 == 0:
            print(f'epoch {iteration}/{nepoch}')
        #for param_groups in model.optimizer.param_groups:
        #    print('lr',param_groups['lr'])
        
        # Evaluation of the model on the validation dataset
        model.eval()
        loss_all = 0
        total_samples = 0
        with torch.no_grad():
            for R_lmb_batch, Z_batch, PhiZ_batch in dataloader_validation:
                if R_init_type == 'MLE':
                    R_batch = Z_batch/PhiZ_batch
                elif R_init_type == 'CP':
                    R_batch = CP_optim_fix(Z_batch, PhiZ_batch, 0.99, 0.99, 50, 500)
                elif R_init_type == 'ones':
                    R_batch = torch.ones_like(Z_batch)

                # Forward pass
                out = model.forward(R_batch, R_lmb_batch, Z_batch, PhiZ_batch)
                total_loss = torch.nn.functional.mse_loss(out, R_lmb_batch)

                current_batch_size = Z_batch.size(-1)
                loss_all = loss_all + total_loss.item()*current_batch_size
                total_samples = total_samples + current_batch_size

            loss_value_validation[iteration] = total_loss.item()
            loss_value_validation_all[iteration] = loss_all/total_samples
            # Save the best model on the validation dataset
            # if loss_value_validation[iteration] < loss_validation:
            #     best_validation_state_dict = copy.deepcopy(model.state_dict())
            #     loss_validation = loss_value_validation[iteration]

        # if iteration == 90000:
        #      for param_group in model.optimizer.param_groups:
        #          param_group['lr'] = 1e-6

                
        model.train()
        iteration = iteration + 1

        # Saving the model in case of a crash
        if iteration%10000 == 0:
            save_path = os.path.join(save_dir, f'save_scheduler{scheduler}_{iteration}.pth')
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'scheduler_state_dict': model.scheduler.state_dict() if model.scheduler else None
        }, save_path)
            
        # if iteration == 10000:
        #      model.grow(step = 80)
        #      print(model.active_neurones)
        # if iteration == 20000:
        #      model.grow(step = 100)
        #      print(model.active_neurones)
            
    model_best = copy.deepcopy(model)
    # with torch.no_grad():
    #     model_best.load_state_dict(best_state_dict)
    model_best_validation = copy.deepcopy(model)
    # with torch.no_grad():
    #     model_best_validation.load_state_dict(best_validation_state_dict)

    return model, model_best, model_best_validation, out.detach().cpu().numpy(), loss_value_train[:iteration], loss_value_train_all[:iteration], loss_value_validation[:iteration], loss_value_validation_all[:iteration], dKL_value[:iteration], metric_value[:iteration], norm_value, norm_value_list[:iteration]