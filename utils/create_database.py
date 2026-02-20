import os
import torch
from torch.utils.data import Dataset
import pickle


class CustomDataset(Dataset):

    """
    Custom Dataset:
    For a given folder directory, create the associated database. 
    The folder is organized as follow:
        - File with name : data_{country}_70weeks
        - In each file: daily count Z, associated global infectiousness PhiZ, list of regularisation parameters,
            list of reproduction numbers R obtained with Chambolle-Pock algorithm 
            and the list of regularisation parameters
    We iterate for each file in the folder, and we extract Z, PhiZ and R
    """
    
    def _pre_process(self, Z, PhiZ, R_lmb):
        Z = torch.tensor(Z, dtype=torch.float, device=self.device).unsqueeze(0)
        PhiZ = torch.tensor(PhiZ, dtype=torch.float, device=self.device).unsqueeze(0)
        R_lmb = torch.tensor(R_lmb, dtype=torch.float, device=self.device).unsqueeze(0)

        if self.norm_type == 'max':
            Z_norm = Z/torch.max(Z)
            PhiZ_norm = PhiZ/torch.max(Z)

        elif self.norm_type == 'std':
            std_Z = Z.std(dim=None, keepdim=True, unbiased=False)
            Z_norm = Z/std_Z
            PhiZ_norm = PhiZ/std_Z

        return Z_norm, PhiZ_norm, R_lmb
    

    def __init__(self, dataset, transform = None, device = 'cpu', norm_type = 'std'):

        self.liste_Z = []
        self.liste_PhiZ = []
        self.liste_R = []
        self.transform = transform
        self.device = device
        self.norm_type = norm_type
        
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

                Z_norm, PhiZ_norm, R_lmb = self._pre_process(Z, PhiZ, R_lmb)
                self.liste_R.append(R_lmb)
                self.liste_PhiZ.append(PhiZ_norm)
                self.liste_Z.append(Z_norm)
                
        elif isinstance(dataset, dict):
            dico_dataset = dataset
            for c, (territory, synthetic_data) in enumerate(dico_dataset.items()):
                N_replica = len(synthetic_data)
                for i in range(N_replica):

                    Z = synthetic_data[i]['Z']
                    PhiZ = synthetic_data[i]['ZPhi']
                    R_lmb = synthetic_data[i]['R']

                    Z_norm, PhiZ_norm, R_lmb = self._pre_process(Z, PhiZ, R_lmb)
                    self.liste_R.append(R_lmb)
                    self.liste_PhiZ.append(PhiZ_norm)
                    self.liste_Z.append(Z_norm)

        else:
            raise Exception("Unrecognised dataset format ")
        
        
    def __len__(self):
        return len(self.liste_R)
    
    def __getitem__(self, index):

        Z_norm = self.liste_Z[index]
        PhiZ_norm = self.liste_PhiZ[index]
        R = self.liste_R[index]

        if self.transform:
            Z_norm = self.transform(Z_norm)
            PhiZ_norm = self.transform(PhiZ_norm)
            R = self.transform(R)

        
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

        return R, Z_norm, PhiZ_norm