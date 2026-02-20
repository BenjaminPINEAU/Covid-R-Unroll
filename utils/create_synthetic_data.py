# Import libraries
import numpy as np
import pandas as pd
import re
import os
import pickle
import os
from contextlib import redirect_stdout, redirect_stderr

from include.load_data.get_counts import get_real_counts
from RL_estim.R_univartiate_wOutliers import Rt_U_O
from include.build_synth import buildData_from_countries as generator
from include.optim_tools import crafting_phi
from include.load_data import date_choice


def silent_call(func, *args, **kwargs):
    with open(os.devnull, 'w') as f:
        with redirect_stdout(f), redirect_stderr(f):
            return func(*args, **kwargs)


def extract_config(dataset_type):
    """
    INPUT:
    - dataset_type: train or test
    
    OUTPUT:
    Return a configuration settings dictionary 
    to create synthetic data which contains the keys :
    - territories: list of all the territories in the CSV
    - dates: list of all the date for each territory
    - firstDays: list of all the first day of the time series for each territory
    - lastDays: list of all the last day of the time series for each territory
    - R_by_cluster: list of reproduction number associated to each territory between firstDay and lastDay
    - Z_by_cluster: list of new infections associated to each territory between firstDay and lastDay
    - O_by_cluster: list of outlier associated to each territory between firstDay and lastDay (see EpiJointEstim toolbox)
    - delta_by_cluster: list of scale to generate synthetic data associated to each territory between firstDay and lastDay
    """

    # Initialization
    Z_by_cluster = []
    delta_by_cluster = []
    dates = []
    lambdaU_L = 3.5
    lambdaU_O = 0.03

    # Extract data from the CSV file
    df = pd.read_csv(f'../data/datasets/{dataset_type}.csv')
    territories = np.array(df['Country'])
    firstDays = np.array(df['firstDay'])
    lastDays = np.array(df['lastDay'])

    dataset_size = len(territories)

    # Iterate on each territory
    for i in range(dataset_size):
        c = re.sub(r"\d.*$", "", territories[i]).strip()

        # get the real counts between firstDay and lastDay
        ZData, options = silent_call(get_real_counts, c, firstDays[i], lastDays[i], 'JHU', "../")
        delta_by_cluster.append(np.std(ZData))
        dates.append(options['dates'])
        Z_by_cluster.append(ZData)

    # Compute the reproduction number and the outliers between firstDay and lastDay for all territories
    R_by_cluster, O_by_cluster, _ = silent_call(Rt_U_O, Z_by_cluster, lambdaU_L, lambdaU_O, options=options)
    config = {
        'territories' : territories,
        'dates' : dates,
        'firstDays' : firstDays,
        'lastDays' : lastDays, 
        'R_by_cluster' : np.array(R_by_cluster),
        'Z_by_cluster' : np.array(Z_by_cluster),
        'O_by_cluster' : np.array(O_by_cluster),
        'delta_by_cluster' : np.array(delta_by_cluster)
        }
    
    return config


def create_synthetic_data(dataset_type, N_replica, dico_omega, save = False, save_path = None):
    """
    INPUT:
    - dataset_type: train or test
    - N_repica: number of synthetic data to generate from each reproduction number
    - dico_omega: dictionary containing the variance level
    - save: bool to save dataset or not
    - save_path: path where the dataset is saved

    OUTPUT:
    - dico_Zsynth: dictionary containting 
    """

    if not save and save_path:
        print('Warning: got path for saving but save is False.')
        print('Nothing will be saved.')

    print(f'Extracting information from {dataset_type}.csv.')
    config = extract_config(dataset_type)

    nb_territories = len(config['territories'])
    cluster_sizes = [1 for _ in range(len(config['territories']))]

    dico_Zsynth = {}
    
    for _, (param_name, param_omega) in enumerate(dico_omega.items()):
        print(fr'Creating dataset for variance level omega = {param_omega}')
        nb_generation_tot = N_replica*nb_territories
        nb_generation = 0
        niter = 0
        nmax = 30000
        dico_Zsynth[param_name] = {config['territories'][i] : [] for i in range(nb_territories)}
        omega = param_omega

        while nb_generation < nb_generation_tot and niter < nmax:
            ZData_by_territory, extra = generator.drawZ_multi(cluster_sizes, config['R_by_cluster'], config['O_by_cluster'], config['Z_by_cluster'], with_O=False, firstDay="2022-01-01", gamma=omega*config['delta_by_cluster'], alpha=None)
            for i in range(nb_territories):
                if not np.any(np.convolve(ZData_by_territory[i] == 0, np.ones(3, dtype=int), mode='valid') == 3) and len(dico_Zsynth[param_name][config['territories'][i]]) < N_replica:
                    dico_Zsynth[param_name][config['territories'][i]].append(ZData_by_territory[i])
                    nb_generation += 1
            niter += 1

    dico_datasets = {}

    for _, (param_name, dico_Zsynth_territories) in enumerate(dico_Zsynth.items()):
        dico_datasets[param_name] = {}
        for c, (territory, liste_Zsynth) in enumerate(dico_Zsynth_territories.items()):
            dico_datasets[param_name][f'Territory{c}'] = []
            for i in range(N_replica):
                Z = liste_Zsynth[i]
                dates_crop, ZDataCropped = date_choice.cropDatesPlusOne(config['firstDays'][c], config['lastDays'][c], config['dates'][c], Z)
                Phi = crafting_phi.buildPhi()
                tmpDates, ZDataDep, ZPhiDep = crafting_phi.buildZPhi(dates_crop, ZDataCropped, Phi)
                data = {
                    "Z_origin" : config['Z_by_cluster'][c],
                    "Z" : ZDataDep,
                    "ZPhi" : ZPhiDep,
                    "R" : config['R_by_cluster'][c], 
                    "dates" : dates_crop,
                    "lambdaU_L" : 3.5, 
                    "lambdaU_O" : 0.03, 
                    "omega" : dico_omega[param_name],
                    "N_replica" : N_replica,
                    "delta" : config['delta_by_cluster'][c],
                }
                dico_datasets[param_name][f'Territory{c}'].append(data)

                if save: 
                    new_folder_path = os.path.join(save_path, f"varlevel{param_name}")
                    os.makedirs(new_folder_path, exist_ok=True)
                    new_file_path = os.path.join(new_folder_path, f'groundTruthR{c}')

                    new_file_path_num = new_file_path + f"_realization{i}.pickle"
            
                    with open(new_file_path_num, 'wb') as handle:
                        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dico_datasets


def create_multiple_variance_level_dataset(dico_datasets, save = False, save_path = None):

    if not save and save_path:
        print('Warning: got path for saving but save is False.')
        print('Nothing will be saved.')

    if save:
        new_folder_path = os.path.join(save_path, f"varlevel_all")
        os.makedirs(new_folder_path, exist_ok=True)

    dico_dataset = {}
    for _, (param_name, dico_territories) in enumerate(dico_datasets.items()):
        for c, (territory, dico_territory) in enumerate(dico_territories.items()):
            data = dico_datasets[param_name][f'Territory{c}']
            dico_dataset[f'Territory{c}_omega{param_name}'] = data
            N_replica = len(data)
            if save:
                for i in range(N_replica):
                    new_file_path = os.path.join(new_folder_path, f'Territory{c}_omega{param_name}')

                    new_file_path_num = new_file_path + f"_n{i}.pickle"
           
                    with open(new_file_path_num, 'wb') as handle:
                        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return dico_dataset