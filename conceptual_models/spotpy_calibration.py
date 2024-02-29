# ## Code to calibrate a process-based hydrological model


# **Description**
# 
# The following notebook contains the code to create, train, and test a process-based rainfall-runoff model. The
# calibration is done using the library [Spotpy](https://doi.org/10.1371/journal.pone.0145180)[1]. When multiple basins
# are being calibrated, the code make the calibration in parallel (one basin per core) to speed up the computation.
# 
# **Authors:**
# - Eduardo Acuna Espinoza (eduardo.espinoza@kit.edu)
# 
# **References:**
# [1]: "Houska, T., Kraft, P., Chamorro-Chavez, A. and Breuer, L.: SPOTting Model Parameters Using a Ready-Made Python Package, PLoS ONE, 10(12), e0145180, doi:10.1371/journal.pone.0145180, 2015"


#Import necessary packages
import os
import pandas as pd
import numpy as np
import multiprocessing
import time
from typing import List, Dict, Tuple

import spotpy
from spotpy.objectivefunctions import rmse as obj_func # objective function used during optimization
#from spotpy.likelihoods import gaussianLikelihoodMeasErrorOut as obj_func

from hydrological_models import bucket as hydrological_model # define which model will be calibrated
from calibration_methods import sce as calibration_method


# Initialize information
path_entities= '../data/basin_id/basins_camels_gb_60.txt'
path_ts = '../data/CAMELS_GB/timeseries/'
warmup = 365
forcing_variables = ['precipitation', 'peti', 'temperature']
target_variables = ['discharge_spec']
training_period = ['1980-10-01','1997-12-31']
testing_period = ['1997-01-01','2008-12-31']

# Save results
path_output = '../results/models/conceptual_models/'

# use when one select the best parameters, depends on the loss function one wants. 
maximize = False # True for gaussian_likelihood, False of rmse

# Set seed to have reproducible results
np.random.seed(42)

# Read information
selected_basins_id= np.loadtxt(path_entities, dtype='str').tolist() 
# Check if the path where one will store the results exists. In case it does not, it creates such path.
if not os.path.exists(path_output):
    # Create the folder
    os.makedirs(path_output)
    print(f"Folder '{path_output}' created successfully.")
else:
    print(f"Folder '{path_output}' already exists.")


class calibration_object(object):
    """Create a calibration object following the spotpy library[#]_.

    Parameters
    ----------
    model : 
        Entity of the class that define the hydrological model that will be calibrated
    input: np.ndarray
        Input to run the model during the calibration period
    target: np.ndarray
        Observed values used as target during the calibration process
    obj_func : 
        loss function used during optimization
    warmup: int
        number of time intervals before the results are used to calculate the loss function

    References
    ----------
    .. [#] Houska, T., Kraft, P., Chamorro-Chavez, A. and Breuer, L.: SPOTting Model Parameters Using a Ready-Made 
    Python Package, PLoS ONE, 10(12), e0145180, doi:10.1371/journal.pone.0145180, 2015
    """
    
    def __init__(self, model, input: np.ndarray, target: np.ndarray, obj_func, warmup: int=0):      
        # model that will be optimized and parameters that will be optimized
        self.model = model
        self.params = self._init_optimization_parameters(model.parameter_ranges)

        # Information that will be used during optimization
        self.inputs= input
        self.target = target

        # mask to avoid nan in target and buffer for warmup period
        self.mask = ~np.isnan(self.target)
        self.warmup=warmup

        #objective function for optimization
        self.obj_func = obj_func
        
    def _init_optimization_parameters(self, parameter_ranges: Dict[str, List[float]]) -> List:
        """Create a list to define the optimization parameters so spotpy recognize them correctly

        Parameters
        ----------
        parameter_ranges: Dict[str, List[float]]
            Dictionary where the keys are the name of the calibration parameters and the values are the range in which
            the parameter can vary

        Returns
        -------
        parameter_list: List
            List with the parameters that will be optimized
        """  
        parameter_list = []
        for param_name, param_range in parameter_ranges.items():
            parameter_list.append(spotpy.parameter.Uniform(low=param_range[0], high=param_range[1], name=param_name))
        return parameter_list
    
    def parameters(self):
        return spotpy.parameter.generate(self.params)
    
    def simulation(self, x):
        sim_q, _ = self.model.run_model(self.inputs, x)
        return sim_q[:,0]
    
    def evaluation(self):
        return self.target[:,0]
    
    def objectivefunction(self,simulation,evaluation):
        masked_evaluation = evaluation[self.warmup:][self.mask[self.warmup:,0]]
        masked_simulation = simulation[self.warmup:][self.mask[self.warmup:,0]]

        like = self.obj_func(masked_evaluation,masked_simulation)
        return like


def read_information(path_ts: str, basin_id: str, forcing: List[str], target: List[str], time_period: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a calibration object following the spotpy library[#]_.
    Parameters
    ----------
    path_ts : str
        path to the folder were the time series of the different basins are stored
    basin_id : str
        Id of the basin on interest
    forcings : List[str]
        Name of the variables that will be used as forcings
    target : List[str]
        Name of the variables that will be used as target
    time_period : List[str]
        initial and final date (e.g. ['1987-10-01','1999-09-30']) of the time period of interest 

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
            - inputs: pd.DataFrame
                Dataframe with the catchment input timeseries
            - target: pd.DataFrame
                Dataframe with the catchment target timeseries  

    """
    
    # Read information that will be used to optimize the model
    path_timeseries = path_ts + 'CAMELS_GB_hydromet_timeseries_'+ basin_id+ '_19701001-20150930.csv'
    df = pd.read_csv(path_timeseries)
    df = df.set_index('date')
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
    
    # forcings
    df_forcing = df.loc[time_period[0]:time_period[1], forcing]
    # target
    df_target = df.loc[time_period[0]:time_period[1], target]

    # save information
    inputs= df_forcing.to_numpy()
    target = df_target.to_numpy().reshape((-1,1))

    return inputs, target


def NSE_loss(evaluation:np.ndarray, simulation:np.ndarray) -> np.ndarray:
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
    return np.round(1.0-nse_loss,2)


def calibrate_basin(path_ts: str, basin: str, forcing_variables: List[str], target_variables: List[str], 
                    training_period: List[str], obj_func, warmup: int, path_output: str):
    
    """Function to run the calibration

    All the process is encompassed on a single function so we can run the different basins in parallel, one basin
    per core

    Parameters
    ----------
    path_ts : str
        path to the folder were the time series of the different basins are stored
    basin_id : str
        Id of the basin on interest
    forcings : List[str]
        Name of the variables that will be used as forcings
    target : List[str]
        Name of the variables that will be used as target
    training_period : List[str]
        initial and final date (e.g. ['1987-10-01','1999-09-30']) of the time period of interest
    obj_func : 
        loss function used during optimization
    warmup: int
        number of time intervals before the results are used to calculate the loss function
    path_output : str
        path to the folder where the results will be stored
    """
    # Initialize instance of model and calibration method
    hyd_model = hydrological_model()
    optimizer = calibration_method()
      
    # Read information that will be used during optimization
    input, target = read_information(path_ts = path_ts, 
                                     basin_id = basin, 
                                     forcing = forcing_variables, 
                                     target = target_variables, 
                                     time_period = training_period)
    
    #Create setup object
    calibration_obj = calibration_object(model= hyd_model, 
                                         input = input,
                                         target = target,
                                         obj_func=obj_func,
                                         warmup=warmup)
    
    sampler = optimizer.run_calibration(calibration_obj=calibration_obj, 
                                        path_output= path_output, 
                                        file_id=basin)


if __name__ == '__main__': 
    # Run the calibration of the different basins in parallel ---------------------------------------------------------
    start_time = time.time()
    num_processes = multiprocessing.cpu_count()
    manager = multiprocessing.Manager()

    processes = []
    for basin in selected_basins_id:
        process = multiprocessing.Process(target=calibrate_basin, args=(path_ts, basin, forcing_variables, 
                                                                        target_variables, training_period, 
                                                                        obj_func, warmup, path_output))
        processes.append(process)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    # Calculate and print the calibration time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total calibration time: {int(elapsed_time)} seconds")
    
    # Process and summarize the results ------------------------------------------------------------------------------
    hyd_model = hydrological_model()
    optimizer = calibration_method()
    df_calibration = pd.DataFrame(index=range(len(selected_basins_id)), 
                                  columns=['basin_id', 'NSE_training'] + list(hyd_model.parameter_ranges) + ['NSE_testing'])
    
    # go through all the basins
    for i, basin in enumerate(selected_basins_id):
        # extract calibrated parameters
        file_name = path_output + hyd_model.name + '_' + optimizer.name + '_' + str(basin)
        results = spotpy.analyser.load_csv_results(file_name)
        param = list(spotpy.analyser.get_best_parameterset(results, maximize=maximize)[0])

        # run the model for training
        input, target = read_information(path_ts = path_ts, 
                                         basin_id = basin, 
                                         forcing = forcing_variables, 
                                         target = target_variables, 
                                         time_period = training_period)

        q_sim, _ = hyd_model.run_model(input, param) 
        NSE_training = NSE_loss(evaluation=target[warmup:][~np.isnan(target)[warmup:,0]].flatten(),
                                simulation=q_sim[warmup:][~np.isnan(target)[warmup:,0]].flatten())
    
        # run the model for testing
        input, target = read_information(path_ts = path_ts, 
                                         basin_id = basin, 
                                         forcing = forcing_variables, 
                                         target = target_variables, 
                                         time_period = testing_period)

        q_sim, _ = hyd_model.run_model(input, param) 
        NSE_testing= NSE_loss(evaluation=target[warmup:][~np.isnan(target)[warmup:,0]].flatten(),
                              simulation=q_sim[warmup:][~np.isnan(target)[warmup:,0]].flatten())
        
        # Save the result of the basin
        df_calibration.loc[i] = [basin, NSE_training] + param + [NSE_testing]
    
    # Save the results
    df_calibration.to_csv(path_output+hyd_model.name+'_'+optimizer.name+'_summary.csv', index=False)