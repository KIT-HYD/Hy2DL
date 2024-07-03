#Import necessary packages
import numpy as np


def nse_loss(evaluation:np.ndarray, simulation:np.ndarray) -> np.ndarray:
    """Nash--Sutcliffe Efficiency.
    Parameters
    ----------
    evaluation : np.ndarray
        Observed values
    simulation : np.ndarray
        Simulated values

    Returns
    -------
    loss: np.array

    """
    nse_loss = np.sum((simulation - evaluation)**2) / np.sum((evaluation - np.mean(evaluation))**2)
    return np.round(1.0-nse_loss,3)
