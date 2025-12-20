from scipy.io import loadmat
import numpy as np
import os

def load_data(country, dataset = 'dataset'):

    """
    Extract features from a .mat file
    INPUT: 
    - country: the country we want to extract the features
    - lambda_: regularisation factor
    - time_period: desired time period between '1year' and '70weeks'
    - dataset: path to the dataset folder

    OUTPUT: 
    - Z: daily counts
    - PhiZ: global infectiousness associated to Z
    - R_lmb: reproduction number computed with Chambolle-Pock and the choosen regularisation factor
    - date: list of the dates in the time period
    """
    #load the file
    filenames = [f'{dataset}/data_{country}_70weeks', f'{dataset}_validation_rapport_stage/data_{country}_70weeks', f'{dataset}/data_{country}_70weeks.mat', f'{dataset}_validation_rapport_stage/data_{country}_70weeks.mat']
    file_exist = False
    for fname in filenames:
        if os.path.exists(fname):
            data = loadmat(fname, squeeze_me=True)
            file_exist = True
    if not file_exist:
        raise FileNotFoundError(f"Aucun fichier trouvé pour {country} )")
        
    # extract the index of the chosen penalisation parameter
    liste_lambda = data['liste_lambda']
    index = np.where(liste_lambda == 50)[0][0]

    # extract the data
    Z = data["Z"]
    PhiZ = data['ZPhi']
    R_lmb = data['dico_RU'][f'R_{index}'].squeeze().tolist()
    date = data['dates']

    # pre-treatment
    Z[Z < 0] = 0
    Z[Z == 0] = 1
    PhiZ[PhiZ == 0] = 1e4

    return Z, PhiZ, R_lmb, date