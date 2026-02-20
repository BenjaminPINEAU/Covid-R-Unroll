import sys
from pathlib import Path

sys.path.append('../utils')

from dataclasses import dataclass
from dataclasses import asdict

import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter
from matplotlib.gridspec import GridSpec
mpl.rcParams.update(mpl.rcParamsDefault)
import torch.nn as nn
import time
import os
import pickle
import pandas as pd 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.io import loadmat, savemat
import re
from tqdm import trange
from datetime import datetime
import torch.optim as optim
import glob
from dateutil.relativedelta import relativedelta

from function_dual_primal import prox_DKL, proxL1_norm, proxL1_norm_adj
from oplin import op_lin_conv, op_lin_adj_conv
from sliding_median import sliding_median, sliding_median_numpy
from cp_unfolded import CPunfolded_conv as nn_conv
from oplin import discrete_derivation
from function_dual_primal import CP_optim_fix
from create_database import CustomDataset
from oplin import create_D2, discrete_derivation, discrete_derivation_adj
from create_synthetic_data import create_synthetic_data, create_multiple_variance_level_dataset
from config import TrainConfig
from database_gradient_descent import gradient_descent_database, load_model
from metric import perf_SNR, perf_D1R
import display_data


from include.load_data.get_counts import get_real_counts
from include.optim_tools import crafting_phi
from RL_estim import joint_estimation as je
from RL_estim.R_univartiate_wOutliers import Rt_U_O
from include.build_synth import buildData_from_countries as generator
from RL_estim.R_epiEstim import Rt_Gamma


color_orange = "#FF8C00"  
color_bleu = '#00008B'
color_green = 'forestgreen'
color_green_cori = [0, 0.5, 0]

x_pos = np.arange(4)
y_ticks_filters = ['5','10','20', '25']
x_ticks_layers = ['5','10','15', '20']

mpl.rcParams["xtick.labelsize"] = 50
mpl.rcParams["ytick.labelsize"] = 50
mpl.rcParams["axes.titlesize"] = 50
mpl.rcParams["lines.linewidth"] = 4
plt.rc("axes", labelsize=42.5)
plt.rc("legend", fontsize=50)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
mpl.rcParams["font.family"] = "roman"
