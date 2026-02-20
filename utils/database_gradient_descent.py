from torch.utils.data import DataLoader
import torch
import os
from tqdm import trange
from scipy.io import savemat
from dataclasses import asdict
import pickle

from cp_unfolded import CPunfolded_conv as nn_conv
from function_dual_primal import CP_optim_fix
from create_database import CustomDataset


def load_model(model, init_param):

    """
    Set the model parameters to certain values

    INPUT:
    - model: the model we use
    - init_param: the value of the parameters to be set

    OUTPUT:
    - the model set with the specifics values
    """

    param_model = torch.load(init_param, map_location="cpu")
    model.load_state_dict(param_model['model_state_dict'], strict = False)
    model.optimizer.load_state_dict(param_model['optimizer_state_dict'])
    

def gradient_descent_database(config):

    """    
    This function operate a gradient descent on the selected network to learn parameters on a database
    and give a prediction of R

    INPUT: 
    - device: cpu or gpu 
    - dataset_train: custom dataset for training
    - nbatch: size of the batch
    - loss_choice: L1 or L2
    - nepoch: number of epoch to learn the parameters
    - K_layer: depth of the network
    - lmb_init: initial value of lambda
    - tau & sigma: convergence parameters of the CP algorithm
    - init_param: file to initialise the parameters of the selected network
    - scheduler: adaptative learning rate strategy
    - learning_rate: initial learning rate of the network
    - optimizer: optimizer for the gradient descent

    OUTPUT:
    - model: final model
    - loss_value_train: loss through the epochs on training dataset
    """

    nbatch = config.nbatch
    nepoch = config.nepoch
    device = config.device
    K_layer = config.K_layer
    learning_rate = config.learning_rate
    init_param = config.init_param
    nb_filter = config.nb_filter
    save = config.save
    save_path = config.save_path 

    dataset_custom = CustomDataset(config.dataset, device = device, norm_type = config.norm_type)

    dataloader_train = DataLoader(dataset_custom, batch_size = nbatch, shuffle = True, num_workers = 0)


    # Initialize parameters
    loss_value_train_all = 1e10 * torch.ones(nepoch) 

    iteration = 0 
    R_init_type = config.R_init_type

    # create the folder to save data during the training
    if not save and save_path:
        print('Warning: got path for saving but save is False.')
        print('Nothing will be saved.')


    # Initialize the model
    model = nn_conv(device, K_layer, learning_rate = learning_rate, nb_filter = nb_filter)

    # Adjust the model settings if necessary
    if init_param:
        load_model(model, init_param)

    # Iteration for each epochs
    model.optimizer.zero_grad()
    
    #while iteration < nepoch:
    for iteration in trange(nepoch, desc='Training', leave=True):
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
            out = model.forward(R_batch, Z_batch, PhiZ_batch)
            total_loss = torch.nn.functional.mse_loss(out, R_lmb_batch)

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
            out = model.forward(R_batch, Z_batch, PhiZ_batch)
            total_loss = torch.nn.functional.mse_loss(out, R_lmb_batch)

            current_batch_size = Z_batch.size(-1)
            loss_all = loss_all + total_loss.item()*current_batch_size
            total_samples = total_samples + current_batch_size

        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm ** 2
        total_norm = total_norm ** 0.5

        loss_value_train_all[iteration] = loss_all/total_samples
        if iteration % round(nepoch*0.1) == 0:
            print('Loss : ', round(loss_value_train_all[iteration].item(), 4))
        
        iteration = iteration + 1

        # Saving the model in case of a crash
        if iteration%round(nepoch*0.1) == 0 and save:
            save_path_temps = os.path.join(save_path, f'save/iter_{iteration}.pth')
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'scheduler_state_dict': model.scheduler.state_dict() if model.scheduler else None
        }, save_path_temps)

    if save:
        os.makedirs(save_path, exist_ok=True)
        # Save the final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
        }, os.path.join(save_path, f'models/model.pth'))

        # Save data
        with torch.no_grad():
            dico = {'R' : out, 'loss_train' : loss_value_train_all[:iteration]}
            new_file_path = os.path.join(save_path, f"results/data.pickle")
            with open(new_file_path, 'wb') as handle:
                    pickle.dump(dico, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Save config
        with open(os.path.join(save_path, f"models/model_exemple_covid_multi_alpha_train_config.txt"), "w") as f:
            for key, value in asdict(config).items():
                f.write(f"{key}: {value}\n")

    return model, loss_value_train_all[:iteration]