#Import necessary packages
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union
from modelcalibration import ModelCalibration

class ModelCalibrationCamelsUS(ModelCalibration):
    
    def __init__(self, 
                 model, 
                 path_data: str,
                 forcing: str,  
                 basin_id: str, 
                 input_variables: List[str], 
                 target_variables: List[str],
                 time_period: Union[None, List[str], Dict[str, str]], 
                 obj_func, 
                 warmup_period: int = 0,
                 path_additional_features: Union[str, None] = None):


        # Run the __init__ method of BaseDataset class, where the data is processed
        self.forcing = forcing
        super(ModelCalibrationCamelsUS, self).__init__(model = model, 
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

        # Input data -----------------------  
        forcing_path = Path(self.path_data) / "basin_mean_forcing" / self.forcing
        file_path = list(forcing_path.glob(f"**/{self.basin_id}_*_forcing_leap.txt"))
        file_path = file_path[0]
        # Read dataframe
        with open(file_path, "r") as fp:
            # load area from header
            fp.readline()
            fp.readline()
            area = int(fp.readline())
            # load the dataframe from the rest of the stream
            df = pd.read_csv(fp, sep=r"\s+")
            df["date"] = pd.to_datetime(df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str), format="%Y/%m/%d")
            df = df.set_index("date")
        
        # Add additional features in case there are any
        if self.additional_features:
            df = pd.concat([df, self.additional_features[self.basin_id]], axis=1)
        
        # Filter variables of interest
        df = df.loc[:, self.input_variables]
        
        # Transform tmax(C) and tmin(C) into tmean(C)
        if 'tmax(C)' in df.columns and 'tmin(C)' in df.columns :
            df['t_mean(C)'] = (df['tmax(C)'] + df['tmin(C)'])/2
            df = df.drop(columns=['tmax(C)', 'tmin(C)'])
            # Filtering the list using list comprehension
            self.input_variables = [item for item in self.input_variables if item not in ['tmax(C)', 'tmin(C)']]
            self.input_variables.append('t_mean(C)')

        # Target data ----------------------
        streamflow_path = Path(self.path_data) / "usgs_streamflow"
        file_path = list(streamflow_path.glob(f'**/{self.basin_id}_streamflow_qc.txt'))
        file_path = file_path[0]
        
        col_names = ["basin", "Year", "Mnth", "Day", "QObs", "flag"]
        df_target = pd.read_csv(file_path, sep=r"\s+", header=None, names=col_names)
        df_target["date"] = pd.to_datetime(df_target.Year.map(str) + "/" + df_target.Mnth.map(str) + "/" + df_target.Day.map(str), format="%Y/%m/%d")
        df_target = df_target.set_index("date")

        # normalize discharge from cubic feet per second to mm per day
        df_target.QObs = 28316846.592 * df_target.QObs * 86400 / (area * 10**6)
        df["QObs(mm/d)"] = df_target.loc[:, "QObs"]
        
        # Filter for specific time period is there is any [list]. If custom time periods are used, what we do it
        # run the model for the whole period and then filter the training/testing subsets.
        if isinstance(self.time_period, list):
            df = df.loc[self.time_period[0]:self.time_period[1], :]
        
        # save information
        timeseries['df'] = df
        timeseries['inputs']= df.loc[:, self.input_variables].to_numpy()
        timeseries['target'] = df.loc[:, self.target_variables].to_numpy().reshape((-1,1))
        
        return timeseries


def check_basins_camelsus(path_data: str, 
                          basins_id: List[str], 
                          training_period: Union[List[str], str], 
                          testing_period: Union[List[str], str]) -> List[str]:
    
    """Check if the basin have target information in the periods of interest

    Parameters
    ----------
    path_data : str
        path to the folder were the time series of the different basins are stored
    basins_id : List[str]
        Ids of the basins
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
        # Target data ----------------------
        streamflow_path = Path(path_data) / "usgs_streamflow"
        file_path = list(streamflow_path.glob(f'**/{basin}_streamflow_qc.txt'))
        file_path = file_path[0]
        
        col_names = ["basin", "Year", "Mnth", "Day", "QObs", "flag"]
        df = pd.read_csv(file_path, sep=r"\s+", header=None, names=col_names)
        df["date"] = pd.to_datetime(df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str), format="%Y/%m/%d")
        df = df.set_index("date")

        # Training period
        if isinstance(training_period, list):
            df_training = df.loc[training_period[0]:training_period[1], "QObs"]
        elif isinstance(training_period, str):
            custom_period = ModelCalibration.create_custom_periods(custom_periods = training_period, basin_id=basin)
            data_split = df.index.isin(custom_period['date'])
            df_training = df.loc[data_split , "QObs"]

        # Testing period
        if isinstance(testing_period, list):
            df_testing = df.loc[testing_period[0]:testing_period[1], "QObs"]
        elif isinstance(testing_period, str):
            custom_period = ModelCalibration.create_custom_periods(custom_periods = testing_period, basin_id=basin)
            data_split = df.index.isin(custom_period['date'])
            df_testing = df.loc[data_split , "QObs"]

        # Check if there are valid target values in training and testing
        if not df_training.isna().all().item() and not df_testing.isna().all().item():
            selected_basins_id.append(basin)

    return selected_basins_id

