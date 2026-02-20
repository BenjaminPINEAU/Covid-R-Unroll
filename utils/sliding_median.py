import numpy as np
import torch


def mad_numpy(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
    """
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def sliding_median_numpy(data, alpha = 0.1, nbDaysMed = 14):
    """
    Computes smoothened data with a sliding median using previous function mad.
    For exemple if alpha = 0, for each t, Z^{denoised}_t = MAD(Z_{t-width,..., t+width}).
    NON-CAUSAL VERSION.
    -param data: ndarray of shape (days,) in MATLAB format
    -param alpha: threshold for denoising. Explicitly alpha =max(|data_t - median(Z_{[t-w, t+w]})|) with w = nbDaysMed/2
    -param nbDaysMed: number of days on which this function is applied
    -return: data, data
    """

    totalDays = len(data)
    width = int((nbDaysMed - 1) / 2)
    dataMed = np.zeros(np.shape(data))

    dataMedians = np.zeros(np.shape(data))
    dataMADs = np.zeros(np.shape(data))
    for k in range(0, totalDays):
        dataWindowed = data[max(k - width, 0): min(k + width + 1, totalDays)]
        dataWindowedMedian = np.median(dataWindowed)
        dataWindowedMAD = mad_numpy(dataWindowed)

        dataMedians[k] = dataWindowedMedian
        dataMADs[k] = dataWindowedMAD

        if abs(data[k] - dataWindowedMedian) > alpha * dataWindowedMAD:
            dataMed[k] = dataWindowedMedian
        else:
            dataMed[k] = data[k]

    return dataMed


"""
adaptation of the previous functions for PyTorch
"""

def mad(x):
    """Median Absolute Deviation (MAD) for PyTorch tensor."""
    median = x.median()
    return (x - median).abs().median()

def sliding_median(data, alpha=0.1, nbDaysMed=14):
    """
    sliding_median function for Pytorch tensor
    """
    
    totalDays = data.shape[-1]
    width = (nbDaysMed - 1) // 2
    device = data.device

    dataMed = torch.zeros_like(data, device = device)
    dataMedians = torch.zeros_like(data, device = device)
    dataMADs = torch.zeros_like(data, device = device)

    for k in range(totalDays):
        start = max(k - width, 0)
        end = min(k + width + 1, totalDays)
        dataWindowed = data[:, start:end]

        dataWindowedMedian = dataWindowed.median()
        dataWindowedMAD = mad(dataWindowed)

        dataMedians[:, k] = dataWindowedMedian
        dataMADs[:, k] = dataWindowedMAD

        if torch.abs(data[:, k] - dataWindowedMedian) > alpha * dataWindowedMAD:
            dataMed[:, k] = dataWindowedMedian
        else:
            dataMed[:, k] = data[:, k]

    return dataMed
