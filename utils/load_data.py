from scipy.io import loadmat
import os
import glob

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
    pattern = os.path.join(dataset, f"data_{country}*.mat")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No file found for {country}")
    files.sort()
    data = loadmat(files[0], squeeze_me=True)
        
    # extract the data
    Z = data["Z"]
    PhiZ = data['ZPhi']
    R_lmb = data['R']
    date = data['dates']

    # pre-treatment
    Z[Z < 0] = 0
    Z[Z == 0] = 1
    PhiZ[PhiZ == 0] = 1e4

    return Z, PhiZ, R_lmb, date