import torch
import numpy as np
import pandas as pd
from typing import Dict


def nse_basin_averaged_loss(y_sim: torch.tensor, y_obs: torch.tensor, 
                            per_basin_target_std: torch.tensor) -> torch.Tensor:
    """Basin-averaged Nash--Sutcliffe Efficiency.

    Loss function where the squared errors are weighed by the std of each basin. A description of this function is 
    available at [#]_.

    Parameters
    ----------
    y_sim : torch.Tensor
        simulated discharges.
    y_obs : torch.Tensor
        observed discharges.
    per_basin_target_std : torch.Tensor
        Standard deviation of the discharge (during training period) for the respective basins. 
    
    Returns
    -------
    loss: torch.Tensor
        value of the basin-averaged NSE

    References
    ----------
    .. [#] Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., and Nearing, G.: "Towards learning
       universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets"
       *Hydrology and Earth System Sciences*, 2019, 23, 5089-5110, doi:10.5194/hess-23-5089-2019
    """
        
    # calculate mask to avoid nan in observation to affect the loss
    mask = ~torch.isnan(torch.flatten(y_obs))
    y_sim_masked = torch.flatten(y_sim)[mask]
    y_obs_masked = torch.flatten(y_obs)[mask]
    basin_std_masked = torch.flatten(per_basin_target_std)[mask]

    squared_error = (y_sim_masked - y_obs_masked)**2
    weights = 1 / (basin_std_masked + 0.1)**2 #The 0.1 is a small constant for numerical stability
    loss = weights * squared_error
    
    return torch.mean(loss)


def nse_loss(df_results: Dict[str, pd.DataFrame], average:bool=True)-> np.array:
    """ Nash--Sutcliffe Efficiency.

    Parameters
    ----------
    df_results : Dict[str, pd.DataFrame]
        Dictionary, where each key is associated with a basin_id and each item is a pandas DataFrame.
        Each dataframe should contained at least two columns: y_sim for the simulated values and y_obs
        for the observed values.
    average : bool
        True if one wants to average the NSE over all the basin (items of the dictionary), or False
        if one wants the value for each one
    
    Returns
    -------
    loss: np.array
        If average==True returns one value for all basins. If average==False returns the NSE for each
        element.
        
    """
    loss=[]
    # Go for each element (basin) of the dictionary
    for basin in df_results.values():
        # Read values
        y_sim = basin['y_sim'].values 
        y_obs = basin['y_obs'].values
        
        # Mask values based on NaN from y_sim (this occurs in validation and testing if there are NaN in the inputs)
        mask_y_sim = ~np.isnan(y_sim)
        y_sim = y_sim[mask_y_sim]
        y_obs = y_obs[mask_y_sim]

        # Mask values based on NaN from y_obs (this occurs in validation and testing if there are NaN in the output)
        mask_y_obs = ~np.isnan(y_obs)
        y_sim = y_sim[mask_y_obs]
        y_obs = y_obs[mask_y_obs]

        # Calculate NSE
        if y_sim.size > 1 and y_obs.size > 1:
            loss.append(1.0 - np.sum((y_sim - y_obs)**2) / np.sum((y_obs - np.mean(y_obs))**2))
        else:
            loss.append(np.nan)
                    
    return np.nanmean(loss) if average else np.asarray(loss)