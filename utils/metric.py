import torch
from utils.oplin import discrete_derivation
from utils.file_selection import build_group_by_pairs, collect_files_by_country, build_group_by_country
from scipy.io import loadmat
import numpy as np
from torchmetrics.audio import SignalNoiseRatio
snr = SignalNoiseRatio()

def metric(oracle, pred):
    """
    Custom metric between an oracle and a prediction
    """
    return torch.sum(torch.abs(oracle - pred)/oracle)

def loss_L1():
    """
    L1 loss
    """
    return torch.nn.L1Loss()

def loss_L2():
    """
    L2 loss
    """
    return torch.nn.MSELoss()


def esp_std_by_country(config, model, countries):
    """
    Compute mean and gaussian confidence interval of predictions for a given list of countries

    INPUT: 
    - config: config Class, see utils.config
    - model: neural network model used for predictions
    - countries: list of string

    OUTPUT: 
    - dico_all: dictionary with mean, standard deviation and gaussian confidence interval for each countries

    """

    # sort the dataset by country
    dataset = config.dataset_validation
    dico = collect_files_by_country(dataset, countries)
    groups_country = build_group_by_country(dico, countries)

    dico_all = {}

    # iterate on each countries
    for group in groups_country:
        
        loss_group = []

        for nfile in group:
            # load data
            file_path = dataset + "/" + nfile
            data = loadmat(file_path, squeeze_me = True)
            liste_lambda = data['liste_lambda']
            index = np.where(liste_lambda == 50)[0][0]

            Z = data["Z"]
            PhiZ = data['ZPhi']
            R_lmb = data['dico_RU'][f'R_{index}'].squeeze().tolist()

            Z[Z < 0] = 0
            Z[Z == 0] = 1
            PhiZ[PhiZ == 0] = 1e4

            # convertion to pytorch tensor
            Z = torch.tensor(Z, dtype=torch.float, device=config.device).unsqueeze(0)
            PhiZ = torch.tensor(PhiZ, dtype=torch.float, device=config.device).unsqueeze(0)
            R_lmb = torch.tensor(R_lmb, dtype=torch.float, device=config.device).unsqueeze(0)

            # normalize the data
            if config.norm_type == 'max':
                Z_norm = Z/torch.max(Z)
                PhiZ_norm = PhiZ/torch.max(Z)

            elif config.norm_type == 'std':
                std_Z = Z.std(dim=None, keepdim=True, unbiased=False)
                Z_norm = Z/std_Z
                PhiZ_norm = PhiZ/std_Z

            # initialize network input
            if config.R_init_type == 'MLE':
                R_batch = Z_norm/PhiZ_norm
            elif config.R_init_type == 'ones':
                R_batch = torch.ones_like(Z_norm)

            # forward pass
            if config.choix_model == "Conv":
                out = model.forward(R_batch.unsqueeze(0), R_lmb.unsqueeze(0), Z_norm.unsqueeze(0), PhiZ_norm.unsqueeze(0)).squeeze(0)
            else:
                out = model.forward(R_batch, R_lmb, Z_norm, PhiZ_norm)

            # compute SNR
            loss = snr(out, R_lmb)
            loss_group.append(loss.item())

        # compute mean, standart deviation and gaussian confidence interval 
        mean = torch.mean(torch.tensor(loss_group))
        std = torch.std(torch.tensor(loss_group))
        dico_group = {'mean' : mean, 'std' : std, 'largeur_gauss' : 1.96/np.sqrt(len(loss_group))*std}
        dico_all[nfile] = dico_group
    
    return dico_all