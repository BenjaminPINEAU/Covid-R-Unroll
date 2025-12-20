import os
import torch
import pandas as pd 
from torch.utils.data import Dataset
from scipy.io import loadmat
import numpy as np
from utils.sliding_median import sliding_median


class CustomDataset(Dataset):

    """
    Custom Dataset:
    For a given folder directory, create the associated database. 
    The folder is organized as follow:
        - File with name : data_{country}_70weeks
        - In each file: daily count Z, associated global infectiousness PhiZ, list of regularisation parameters,
            list of reproduction numbers R obtained with Chambolle-Pock algorithm 
            and the list of regularisation parameters
    We iterate for each country (ie each file) in the folder, and we extract Z, PhiZ and R
    """
    
    def __init__(self, data_dir, transform = None, device = 'cpu', norm_type = 'max'):
        self.liste_Z = []
        self.liste_PhiZ = []
        self.liste_R = []
        self.country = []
        self.transform = transform
        self.device = device
        self.norm_type = norm_type

        print(norm_type)
        
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            if os.path.isfile(filepath):
                data = loadmat(filepath)

            liste_lambda = data['liste_lambda'].squeeze()
            index = np.where(liste_lambda == 50)[0][0]

            Z = data["Z"]
            #Z_alpha = data["Z_alpha"]
            PhiZ = data['ZPhi']
            R_lmb = data['dico_RU'][f'R_{index}'].squeeze().tolist()
            Z[Z < 0] = 0
            Z[Z == 0] = 1
            PhiZ[PhiZ == 0] = 1e4

            Z = torch.tensor(Z, dtype=torch.float, device=self.device)
            PhiZ = torch.tensor(PhiZ, dtype=torch.float, device=self.device)
            R_lmb = torch.tensor(R_lmb, dtype=torch.float, device=self.device)

            if self.norm_type == 'max':
                Z_norm = Z/torch.max(Z)
                PhiZ_norm = PhiZ/torch.max(Z)

            elif self.norm_type == 'std':
                std_Z = Z.std(dim=None, keepdim=True, unbiased=False)
                Z_norm = Z/std_Z
                PhiZ_norm = PhiZ/std_Z

            self.liste_R.append(R_lmb)
            self.liste_PhiZ.append(PhiZ_norm)
            self.liste_Z.append(Z_norm)
            #self.liste_Zalpha.append(Z_alpha)
            self.country.append(filename)


    def __len__(self):
        return len(self.liste_R)
    
    def __getitem__(self, index):
        #Z_alpha = self.liste_Zalpha[index]
        #Z_alpha = torch.tensor(Z_alpha, dtype=torch.float, device=self.device)

        Z_norm = self.liste_Z[index]
        PhiZ_norm = self.liste_PhiZ[index]
        R = self.liste_R[index]

        if self.transform:
            Z_norm = self.transform(Z_norm)
            PhiZ_norm = self.transform(PhiZ_norm)
            R = self.transform(R)

        # print(self.country[index])
        # print(len(R[0]))
        # print(len(Z_norm))
        # print(len(PhiZ))
        
        target_length = 490
        T_Znorm = Z_norm.shape[1]
        T_PhiZnorm = PhiZ_norm.shape[1]
        T_R = R.shape[1]
        
        if T_Znorm > target_length:
            Z_norm = Z_norm[:, :target_length]
        if T_PhiZnorm > target_length:
            PhiZ_norm = PhiZ_norm[:, :target_length]
        if T_R > target_length:
            R = R[:, :target_length]

        return R, Z_norm, PhiZ_norm#, self.country[index]
    


class CustomDataset_window(Dataset):

    """
    Custom Dataset:
    For a given folder directory, create the associated database. 
    The folder is organized as follow:
        - File with name : data_{country}_70weeks
        - In each file: daily count Z, associated global infectiousness PhiZ, list of regularisation parameters,
            list of reproduction numbers R obtained with Chambolle-Pock algorithm 
            and the list of regularisation parameters
    We iterate for each country (ie each file) in the folder, and we extract Z, PhiZ and R
    """
    
    def __init__(self, data_dir, transform = None, device = 'cpu', window = 98):
        self.liste_Z = []
        self.liste_Zalpha = []
        self.liste_PhiZ = []
        self.liste_R = []
        self.country = []
        self.transform = transform
        self.device = device
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            data = loadmat(filepath)

            liste_lambda = data['liste_lambda'].squeeze()
            index = np.where(liste_lambda == 50)[0][0]

            Z = data["Z"][0]
            Z_alpha = data["Z_alpha"][0]
            PhiZ = data['ZPhi'][0]
            R_lmb = data['dico_RU'][f'R_{index}'].squeeze().tolist()[0]
            Z[Z < 0] = 0
            PhiZ[PhiZ == 0] = 1e4

            T = len(Z)

            for start in range(0, T - window + 1, window):
                end = start + window
                z = Z[start:end]
                za = Z_alpha[start:end]
                pz = PhiZ[start:end]
                r = R_lmb[start:end]

                self.liste_Z.append([z])
                self.liste_Zalpha.append([za])
                self.liste_R.append([r])
                self.liste_PhiZ.append([pz])



    def __len__(self):
        return len(self.liste_R)
    
    def __getitem__(self, index):
        Z_alpha = self.liste_Zalpha[index]
        Z_alpha = torch.tensor(Z_alpha, dtype=torch.float, device=self.device)

        Z = torch.tensor(self.liste_Z[index], dtype=torch.float, device=self.device)
        PhiZ = torch.tensor(self.liste_PhiZ[index], dtype=torch.float, device=self.device)
        R = torch.tensor(self.liste_R[index], dtype=torch.float, device=self.device)

        std_Z = Z.std(dim=None, keepdim=True, unbiased=False)
        #Z_norm = Z_alpha/std_Z
        # Z_norm = Z/std_Z
        # PhiZ = PhiZ/std_Z
        Z_norm = Z/torch.max(Z)
        PhiZ = PhiZ/torch.max(PhiZ)

        if self.transform:
            Z_norm = self.transform(Z_norm)
            PhiZ = self.transform(PhiZ)
            R = self.transform(R)

        # print(self.country[index])
        # print(len(R[0]))
        # print(len(Z_norm))
        # print(len(PhiZ))

        return R, Z_norm, PhiZ#, self.country[index]