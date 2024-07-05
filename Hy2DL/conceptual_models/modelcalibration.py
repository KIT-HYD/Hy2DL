import pandas as pd
import numpy
import random
import numpy as np
import spotpy
import pickle
import multiprocessing
from typing import List, Union, Dict

class ModelCalibration(object):
    """Create a calibration object following the spotpy library[#]_.

    Parameters
    ----------

    References
    ----------
    .. [#] Houska, T., Kraft, P., Chamorro-Chavez, A. and Breuer, L.: SPOTting Model Parameters Using a Ready-Made 
    Python Package, PLoS ONE, 10(12), e0145180, doi:10.1371/journal.pone.0145180, 2015
    """

    def __init__(self, 
                 model, 
                 path_data: str,
                 basin_id: str, 
                 input_variables: List[str], 
                 target_variables: List[str],
                 time_period: Union[None, List[str], Dict[str, str]], 
                 obj_func, 
                 warmup_period: int,
                 path_additional_features: Union[str, None]):      
        
        # model and parameters that will be optimized
        self.model = model
        self.params = self._init_optimization_parameters(model.parameter_ranges)

        # Initialize information
        self.path_data = path_data
        self.basin_id = basin_id
        self.input_variables= input_variables
        self.target_variables = target_variables

        self.warmup_period=warmup_period
        if isinstance(time_period, list):
            self.time_period = time_period
        elif isinstance(time_period, str):
            self.time_period = ModelCalibration.create_custom_periods(custom_periods = time_period, basin_id=basin_id)
        else:
            self.time_period = None

        #objective function for optimization
        self.obj_func = obj_func
        
        # read information that will be used during optimization
        if path_additional_features:
            self.additional_features = self._load_additional_features(path_additional_features)
        else:
            self.additional_features =  None
            
        self.timeseries = self._read_data()

        # Initialize vectors to do custom splitting (custom training/testing periods)
        if isinstance(time_period, list): # no custom splitting
            self.data_split = np.full(len(self.timeseries['df']), True, dtype=bool)
        elif isinstance(time_period, str):
             self.data_split = self.timeseries['df'].index.isin(self.time_period['date'])

    
    def parameters(self):
        return spotpy.parameter.generate(self.params)
    
    def simulation(self, x):
        q_sim, _ = self.model.run_model(self.timeseries['inputs'], x)
        return q_sim[:,0]
    
    def evaluation(self):
        return self.timeseries['target'][:,0]
    
    def objectivefunction(self,simulation,evaluation):
        
        evaluation = evaluation[self.warmup_period:][self.data_split[self.warmup_period:]]
        simulation = simulation[self.warmup_period:][self.data_split[self.warmup_period:]]

        # Mask nans from evaluation data
        mask_nans = ~np.isnan(evaluation)
        masked_evaluation = evaluation[mask_nans]
        masked_simulation = simulation[mask_nans]

        like = self.obj_func(masked_evaluation,masked_simulation)
        return like
    
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
    
    def _read_data(self) -> pd.DataFrame:
        raise NotImplementedError
        
    def _load_additional_features(self, path_additional_features) -> Dict[str, pd.DataFrame]:
        """Read pickle dictionary containing additional features.

        Returns
        -------
        additional_features: Dict[str, pd.DataFrame]
            Dictionary where each key is a basin and each value is a date-time indexed pandas DataFrame with the 
            additional features
        """
    
        with open(path_additional_features, "rb") as file:
            additional_features = pickle.load(file)
        return additional_features
    
    @staticmethod
    def create_custom_periods(custom_periods, basin_id):
        with open(custom_periods, 'rb') as f:
            # Load the object from the pickle file
            dict_train= pickle.load(f)

            date_ranges = []
            for i, start_date in enumerate(dict_train[basin_id]['start_dates']):
                date_range = pd.date_range(start_date, dict_train[basin_id]['end_dates'][i])
                date_ranges.append(date_range)
            
            continuous_series = pd.concat([pd.DataFrame(date_range, columns=['date']) for date_range in date_ranges])
            continuous_series = continuous_series.drop_duplicates()
            continuous_series.reset_index(drop=True, inplace=True)
            
            return continuous_series
        

def calibrate_single_basin(calibration_object,
                           optimizer,
                           path_output: str,
                           random_seed:int = 42):
    
    # Set seed to have reproducible results
    random.seed(random_seed)
    np.random.seed(random_seed)
    optimizer.run_calibration(calibration_obj=calibration_object, path_output= path_output)


def calibrate_basins(training_object, optimization_method, basins, path_output: str, random_seed: int = 42):

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    pool.starmap(calibrate_single_basin, [(training_object[basin], optimization_method, path_output, random_seed) 
                                          for basin in basins])

    # Close the pool after processing
    pool.close()
    pool.join()