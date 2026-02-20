import torch
from scipy.io import loadmat
import numpy as np
from torchmetrics.audio import SignalNoiseRatio
snr = SignalNoiseRatio()
import os, pickle
from RL_estim.R_epiEstim import Rt_Gamma


def compute_lossSNR(R_lmb, Z, PhiZ, config, model, data):

    """
    INPUT:
    - R_lmb: R ground truch
    - Z: daily count
    - PhiZ: global infectiousness
    - config: config setting class (see config.py)
    - model: chosen estimator model
    
    OUTPUT:
    - loss: SNR between the ground truth R and the predicted estimator for the chosen model
    """

    Z = torch.tensor(Z[:], dtype=torch.float, device=config.device)
    PhiZ = torch.tensor(PhiZ[:], dtype=torch.float, device=config.device)
    R_lmb = torch.tensor(R_lmb[:], dtype=torch.float, device=config.device)

    if model == 'EpiEstim':
        Rt, _ = Rt_Gamma(np.array(Z), tau=7, display=False, options=data)
        Rt = torch.tensor(Rt)
        R_lmb = R_lmb[1:]
    elif model == 'MLE':
        Rt = Z/PhiZ
            
    else:
        if config.norm_type == 'max':
            Z_norm = Z/torch.max(Z)
            PhiZ_norm = PhiZ/torch.max(Z)
        elif config.norm_type == 'std':
            std_Z = Z.std(dim=None, keepdim=True, unbiased=False)
            Z_norm = Z/std_Z
            PhiZ_norm = PhiZ/std_Z

        if config.R_init_type == 'MLE':
            R_batch = Z_norm/PhiZ_norm
        elif config.R_init_type == 'ones':
            R_batch = torch.ones_like(Z_norm)
            
        Rt = model.forward(R_batch.unsqueeze(0), Z_norm.unsqueeze(0), PhiZ_norm.unsqueeze(0))

    loss = snr(torch.flatten(Rt), R_lmb)
    return loss


def compute_D1R(Z, PhiZ, config, model):

    """
    INPUT:
    - Z: daily count
    - PhiZ: global infectiousness
    - config: config setting class (see config.py)
    - model: chosen estimator model
    
    OUTPUT:
    - D1R: smoothness norm(D1R) of predicted estimator R for the chosen model
    """

    Z = torch.tensor(Z[:], dtype=torch.float, device=config.device)
    PhiZ = torch.tensor(PhiZ[:], dtype=torch.float, device=config.device)

    if config.norm_type == 'max':
        Z_norm = Z/torch.max(Z)
        PhiZ_norm = PhiZ/torch.max(Z)
    elif config.norm_type == 'std':
        std_Z = Z.std(dim=None, keepdim=True, unbiased=False)
        Z_norm = Z/std_Z
        PhiZ_norm = PhiZ/std_Z

    if config.R_init_type == 'MLE':
        R_batch = Z_norm/PhiZ_norm
    elif config.R_init_type == 'ones':
        R_batch = torch.ones_like(Z_norm)
            
    Rt = model.forward(R_batch.unsqueeze(0), Z_norm.unsqueeze(0), PhiZ_norm.unsqueeze(0))
    Rt = torch.flatten(Rt).detach().numpy()
    D1R = np.linalg.norm(Rt[1:] - Rt[:-1], 2)**2
    return D1R


def perf_SNR(config, model):

    """
    INPUT:
    - config: config setting class (see config.py)
    - model: chosen estimator model
    
    OUTPUT:
    - dico_metric: containing the average SNR and standard deviation of the SNR for each time series in a database
    """

    dataset = config.dataset
    list_loss = []

    if isinstance(dataset, str):
        data_dir = dataset
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)

            Z = data["Z"]
            PhiZ = data['ZPhi']
            R_lmb = data['R']

            loss = compute_lossSNR(R_lmb, Z, PhiZ, config, model, data)
            list_loss.append(loss.item())

    elif isinstance(dataset, dict):
        dico_dataset = dataset
        for c, (territory, synthetic_data) in enumerate(dico_dataset.items()):
            N_replica = len(synthetic_data)
            for i in range(N_replica):

                Z = synthetic_data[i]['Z']
                PhiZ = synthetic_data[i]['ZPhi']
                R_lmb = synthetic_data[i]['R']

                loss = compute_lossSNR(R_lmb, Z, PhiZ, config, model, data)
                list_loss.append(loss.item())


    mean = torch.mean(torch.tensor(list_loss))
    std = torch.std(torch.tensor(list_loss))
    dico_metric = {'mean' : mean, 'std' : std, 'gaussian_width' : 1.96/np.sqrt(len(list_loss))*std}

    return dico_metric


def perf_D1R(config, model):

    """
    INPUT:
    - config: config setting class (see config.py)
    - model: chosen estimator model
    
    OUTPUT:
    - D1R_mean: containing the average smoothness norm(D1R) of predicted estimator R for the chosen model, for each time series in a database
    """

    dataset = config.dataset
    list_D1R = []

    if isinstance(dataset, str):
        data_dir = dataset
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
        
            Z = data["Z"]
            PhiZ = data['ZPhi']

            D1R = compute_D1R(Z, PhiZ, config, model, data)
            list_D1R.append(D1R)

    elif isinstance(dataset, dict):
        dico_dataset = dataset
        for c, (territory, synthetic_data) in enumerate(dico_dataset.items()):
            N_replica = len(synthetic_data)
            for i in range(N_replica):

                Z = synthetic_data[i]['Z']
                PhiZ = synthetic_data[i]['ZPhi']

                D1R = compute_D1R(Z, PhiZ, config, model, data)
                list_D1R.append(D1R)

    D1R_mean = np.mean(np.array(list_D1R))

    return D1R_mean