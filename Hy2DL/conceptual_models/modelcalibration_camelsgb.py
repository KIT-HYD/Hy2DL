#Import necessary packages
import pandas as pd
from typing import List, Dict, Union
from modelcalibration import ModelCalibration

class ModelCalibrationCamelsGB(ModelCalibration):
    
    def __init__(self, 
                 model, 
                 path_data: str,
                 basin_id: str, 
                 input_variables: List[str], 
                 target_variables: List[str],
                 time_period: Union[None, List[str], Dict[str, str]], 
                 obj_func, 
                 warmup_period: int = 0,
                 path_additional_features: Union[str, None] = None):


        # Run the __init__ method of BaseDataset class, where the data is processed
        super(ModelCalibrationCamelsGB, self).__init__(model = model, 
                                                       path_data = path_data,  
                                                       basin_id = basin_id, 
                                                       input_variables = input_variables, 
                                                       target_variables= target_variables,
                                                       time_period = time_period, 
                                                       obj_func = obj_func,
                                                       warmup_period = warmup_period,
                                                       path_additional_features = path_additional_features)

    def _read_data(self) -> Dict[str, pd.DataFrame]:
        # Dictionary to store the information
        timeseries = {}

        # Read information that will be used to optimize the model
        path_timeseries = self.path_data + "/timeseries/CAMELS_GB_hydromet_timeseries_"+ self.basin_id+ "_19701001-20150930.csv"
        df = pd.read_csv(path_timeseries)
        df = df.set_index("date")
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
        
        # Add additional features in case there are any
        if self.additional_features:
            df = pd.concat([df, self.additional_features[self.basin_id]], axis=1)
        
        # Filter for specific time period is there is any, otherwise it takes the whole time series
        if isinstance(self.time_period, list):
            df = df.loc[self.time_period[0]:self.time_period[1], :]
        
        # Filter for specific time period is there is any [list]. If custom time periods are used, what we do it
        # run the model for the whole period and then filter the training/testing subsets.
        df = df.loc[:, self.input_variables + self.target_variables]

        # save information
        timeseries['df'] = df
        timeseries['inputs']= df.loc[:, self.input_variables].to_numpy()
        timeseries['target'] = df.loc[:, self.target_variables].to_numpy().reshape((-1,1))

        return timeseries


def check_basins_camelsgb(path_data: str, 
                          basins_id: List[str], 
                          target_variables: List[str], 
                          training_period: Union[List[str], str], 
                          testing_period: Union[List[str], str]) -> List[str]:
    
    """Check if the basin have target information in the periods of interest

    Parameters
    ----------
    path_data : str
        path to the folder were the time series of the different basins are stored
    basins_id : List[str]
        Ids of the basins
    target_variables : List[str]
        Name of the variables that will be used as target
    training_period : List[str]
        initial and final date (e.g. ["1987-10-01","1999-09-30"]) of the training period of interest 
    testing_period : List[str]
        initial and final date (e.g. ["1987-10-01","1999-09-30"]) of the testing period of interest 

    Returns
    -------
    selected_basins_id: List[str]
        valid basins for training and testing

    """   
    selected_basins_id = [] 
    # Check if basin has information in training / testing period
    for basin in basins_id:
        # Read information that will be used to optimize the model
        path_timeseries = path_data + "/timeseries/CAMELS_GB_hydromet_timeseries_"+ basin+ "_19701001-20150930.csv"
        df = pd.read_csv(path_timeseries)
        df = df.set_index("date")
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
              
        # Training period
        if isinstance(training_period, list):
            df_training = df.loc[training_period[0]:training_period[1], target_variables]
        elif isinstance(training_period, str):
            custom_period = ModelCalibration.create_custom_periods(custom_periods = training_period, basin_id=basin)
            data_split = df.index.isin(custom_period['date'])
            df_training = df.loc[data_split , target_variables]
        
        # Testing period
        if isinstance(testing_period, list):
            df_testing = df.loc[testing_period[0]:testing_period[1], target_variables]
        elif isinstance(testing_period, str):
            custom_period = ModelCalibration.create_custom_periods(custom_periods = testing_period, basin_id=basin)
            data_split = df.index.isin(custom_period['date'])
            df_testing = df.loc[data_split , target_variables]

        if not df_training.isna().all().item() and not df_testing.isna().all().item():
            selected_basins_id.append(basin)

    return selected_basins_id