import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from numba import njit, prange


class BaseDataset(Dataset):
    """Base data set class to read and process data.

    This class is inherited by the other subclasses (e.g. CAMELS_US, CAMELS_GB) to read and process the data. The class
    contains all the common operations to that needs to be done independently which database is being used.
    
    This class and its methods were taken from Neural Hydrology [1]_ and adapted for our specific case. 
        
    Parameters
    ----------
    dynamic_input : List[str]
        name of variables used as dynamic series input in the lstm
    target : List[str]
        target variable(s)
    sequence_length : int
        sequence length used for the model
    time_period : List[str]
        initial and final date (e.g. ['1987-10-01','1999-09-30']) of the time period of interest 
    path_data : str
        path to the folder were the data is stored
    path_entities : str
        path to a txt file that contain the id of the entities (e.g. catchment`s ids) that will be analyzed
    entity : str
        id of the entities (e.g. catchment`s id) that will be analyzed. Alternative option to specifying a
        path_entities.
    path_addional features : Optional[str] = None
        Optional parameter. Allows the option to add any arbitrary data that is not included in the standard data sets.
        Path to a pickle file (or list of paths for multiple files), containing a dictionary with each key corresponding 
        to one basin id and the value is a date-time indexed pandas DataFrame.      
    predict_last_n : Optional[int] = 1
        number of timesteps (e.g. days) used to calculate the loss
    static_input : Optional[List[str]] = []
        name of static inputs used as input in the lstm (e.g. catchment attributes)
    conceptual_input : Optional[List[str]] = []
        Optional parameter. We need this when we use hybrid models. Name of variables used as dynamic series input in 
        the conceptual model
    check_NaN : Optional[bool] = True
        Boolean that indicate if one should check of NaN values while processing the data
    
    References
    ----------
    .. [1] F. Kratzert, M. Gauch, G. Nearing and D. Klotz: NeuralHydrology -- A Python library for Deep Learning
        research in hydrology. Journal of Open Source Software, 7, 4050, doi: 10.21105/joss.04050, 2022 
    """
    #Function to initialize the data
    def __init__(self, 
                 dynamic_input: List[str],
                 target: List[str], 
                 sequence_length: int,
                 time_period: List[str],
                 path_data: str,
                 path_entities: str = '',
                 entity: str = '',
                 path_additional_features: Optional[str] = '',
                 predict_last_n: Optional[int] = 1,
                 static_input: Optional[List[str]] = [],
                 conceptual_input: Optional[List[str]] = [],
                 check_NaN:bool = True
                 ):

        self.dynamic_input = dynamic_input 
        self.conceptual_input = conceptual_input
        self.target = target 
        
        self.sequence_length = sequence_length
        self.predict_last_n = predict_last_n
        
        self.path_data = path_data
        self.path_additional_features=path_additional_features
        self.time_period = time_period 
        
        # One can specifiy a txt file with single/multiple entities ids, or directly the entity_id
        if path_entities:
            entities_ids = np.loadtxt(path_entities, dtype='str').tolist() 
            self.entities_ids = [entities_ids] if isinstance(entities_ids, str) else entities_ids # catchments
        elif entity:
            self.entities_ids = [entity]

        # Initialize variables
        self.sequence_data = {} # store information that will be used to run the model
        self.df_ts = {} # store processed dataframes for all basins
        self.scaler = {} # information to standardize the data 
        self.basin_std = {} # std of the target variable of each basin (can be used later in the loss function)
        self.valid_entities= []

        # process the attributes
        self.static_input = static_input # static attributes going as inputs to the lstm
        if static_input:
            self.df_attributes = self._read_attributes()

        # process additional features that will be included in the inputs (optional) ---
        if path_additional_features:
            self.additional_features = self._load_additional_features()
        
        # This loop goes through all the catchments. For each catchment in creates an entry in the dictionary
        # self.sequence_data, where we will store the information that will be sent to the lstm
        for id in self.entities_ids:
            # load time series for specific catchment id
            df_ts = self._read_data(catch_id=id)
            # add additional features (optional)
            if path_additional_features:
                df_ts = pd.concat([df_ts, self.additional_features[id]], axis=1)
            
            # Defines the start date considering the offset due to sequence length. We want that, if possible, the start
            # date is the first date of prediction.
            start_date = pd.to_datetime(self.time_period[0],format="%Y-%m-%d")
            end_date = pd.to_datetime(self.time_period[1],format="%Y-%m-%d")
            freq = pd.infer_freq(df_ts.index)
            warmup_start_date = start_date -\
                (self.sequence_length-self.predict_last_n)*pd.tseries.frequencies.to_offset(freq)
            
            # filter dataframe for the period and variables of interest
            unique_inputs = list(set(dynamic_input+conceptual_input))
            keep_columns = unique_inputs + self.target
            df_ts = df_ts.loc[warmup_start_date:end_date, keep_columns]
            
            # reindex the dataframe to assure continuos data between the start and end date of the time period. Missing 
            # data will be filled with NaN, so this will be taken care of later. 
            full_range = pd.date_range(start=warmup_start_date, end=end_date, freq=freq)
            df_ts = df_ts.reindex(full_range)
            
            # checks for invalid samples due to NaN or insufficient sequence length
            flag = validate_samples(x = df_ts.loc[:, unique_inputs].values, 
                                    y = df_ts.loc[:, self.target].values, 
                                    attributes = self.df_attributes.loc[id].values if static_input else None, 
                                    seq_length = self.sequence_length,
                                    predict_last_n = self.predict_last_n,
                                    check_NaN = check_NaN,
                                    )
            
            # create a list that contain the indexes (basin, day) of the valid samples
            valid_samples = np.argwhere(flag == 1)
            self.valid_entities.extend([(id, int(f[0])) for f in valid_samples])
            
            # store dataframe
            if valid_samples.size>0:
                self.df_ts[id] = df_ts
                # create dictionary entry for the basin
                self.sequence_data[id] = {}
                # store the information of the basin in a nested dictionary
                self.sequence_data[id]['x_d'] = torch.tensor(df_ts.loc[:, self.dynamic_input].values, 
                                                             dtype=torch.float32)
                self.sequence_data[id]['y_obs'] = torch.tensor(df_ts.loc[:, self.target].values, dtype=torch.float32)

                if self.conceptual_input:
                    self.sequence_data[id]['x_conceptual'] = torch.tensor(df_ts.loc[:, self.conceptual_input].values, 
                                                                           dtype=torch.float32)
                if self.static_input:
                    self.sequence_data[id]['x_s'] = torch.tensor(self.df_attributes.loc[id].values, dtype=torch.float32)
                        
    def __len__(self):
        return len(self.valid_entities)
    
    def __getitem__(self, id):
        """Function used by PyTorch's dataloader to extract the information"""
        basin, i = self.valid_entities[id]
        sample = {}
        
        # tensor of inputs data driven part
        x_lstm = self.sequence_data[basin]['x_d'][i-self.sequence_length+1:i+1, :]
        if self.static_input:
            x_s = self.sequence_data[basin]['x_s'].repeat(x_lstm.shape[0],1)
            x_lstm = torch.cat([x_lstm, x_s], dim=1)

        #store data
        sample['x_lstm'] = x_lstm
        sample['y_obs'] = self.sequence_data[basin]['y_obs'][i-self.predict_last_n+1:i+1, :]

        # tensor of inputs conceptual model
        if self.conceptual_input:
            sample['x_conceptual'] = self.sequence_data[basin]['x_conceptual'][i-self.sequence_length+1:i+1, :]

        # optional also return the basin_std
        if self.basin_std:
            sample['basin_std'] = self.basin_std[basin].repeat(sample['y_obs'].size(0)).unsqueeze(1)

        # return information about the basin and the date to facilitate evaluation and ploting of results
        sample['basin'] = basin
        sample['date'] = str(self.df_ts[basin].index[i])
        
        return sample
                
    def _read_attributes(self) -> pd.DataFrame:
        raise NotImplementedError

    def _read_data(self) -> pd.DataFrame:
        raise NotImplementedError

    def _load_additional_features(self) -> Dict[str, pd.DataFrame]:
        """Read pickle dictionary containing additional features.

        Returns
        -------
        additional_features : Dict[str, pd.DataFrame]
            Dictionary where each key is a basin and each value is a date-time indexed pandas DataFrame with the 
            additional features
        """
        with open(self.path_additional_features, "rb") as file:
            additional_features = pickle.load(file)
        return additional_features
  
    def calculate_basin_std(self):
        """Fill the self.basin_std dictionary with the standard deviation of the target variables for each basin"""
        for id, data in self.sequence_data.items():
            self.basin_std[id] = torch.tensor(np.nanstd(data['y_obs'].numpy()), dtype=torch.float32)
    
    def calculate_global_statistics(self, path_save_scaler:Optional[str] = ''):
        """Fill the self.scalar dictionary 
        
        Parameters
        ----------
        path_save_scalar : str
            path to save the scaler as a pickle file 
        
        The function calculates the global mean and standard deviation of the dynamic inputs, target variables and 
        static attributes, and store the in a dictionary. It will be used later to standardize used in the LSTM. This
        function should be called only in training period. 
        """
        global_x = np.vstack([df.loc[:, self.dynamic_input].values for df in self.df_ts.values()])
        self.scaler['x_d_mean'] = torch.tensor(np.nanmean(global_x, axis=0), dtype=torch.float32)
        self.scaler['x_d_std'] = torch.tensor(np.nanstd(global_x, axis=0), dtype=torch.float32)
        del global_x

        global_y = np.vstack([df.loc[:, self.target].values for df in self.df_ts.values()])
        self.scaler['y_mean'] = torch.tensor(np.nanmean(global_y, axis=0), dtype=torch.float32)
        self.scaler['y_std'] = torch.tensor(np.nanstd(global_y, axis=0), dtype=torch.float32)
        del global_y
        
        if self.static_input:
            self.scaler['x_s_mean'] = torch.tensor(self.df_attributes.mean().values, dtype= torch.float32)
            self.scaler['x_s_std'] = torch.tensor(self.df_attributes.std().values, dtype= torch.float32)

        if path_save_scaler: #save the results in a pickle file
            with open(path_save_scaler+'/scaler.pickle', 'wb') as f:
                pickle.dump(self.scaler, f)
    
    def standardize_data(self, standardize_output:bool=True):
        """Standardize the data. 

        The function standardize the data contained in the self.sequence_data dictionary 
        
        Parameters
        ----------
        standardize_output : bool
            Boolean to define if the output should be standardize or not. 
        """
        for basin in self.sequence_data.values():
            # Standardize lstm
            basin['x_d'] = (basin['x_d'] - self.scaler['x_d_mean']) / self.scaler['x_d_std']
            if self.static_input:
                basin['x_s'] = (basin['x_s'] - self.scaler['x_s_mean']) / self.scaler['x_s_std']
            if standardize_output:
                basin['y_obs'] = (basin['y_obs'] - self.scaler['y_mean']) / self.scaler['y_std']


@njit()
def validate_samples(x: np.ndarray, y: np.ndarray, attributes: np.ndarray, seq_length: int, predict_last_n:int=1, 
                     check_NaN:bool=True) -> np.ndarray:
    
    """Checks for invalid samples due to NaN or insufficient sequence length.

    This function was taken from Neural Hydrology [1]_ and adapted for our specific case. 
        
    Parameters
    ----------
    x : np.ndarray
        array of dynamic input;
    y : np.ndarray
        arry of target values;
    attributes : np.ndarray
        array containing the static attributes;
    seq_length : int
        Sequence lengths; one entry per frequency
    predict_last_n: int
        Number of values that want to be used to calculate the loss
    check_NaN : bool
        Boolean to specify if Nan should be checked or not

    Returns
    -------
    flag:np.ndarray 
        Array has a value of 1 for valid samples and a value of 0 for invalid samples.

    References
    ----------
    .. [1] F. Kratzert, M. Gauch, G. Nearing and D. Klotz: NeuralHydrology -- A Python library for Deep Learning
        research in hydrology. Journal of Open Source Software, 7, 4050, doi: 10.21105/joss.04050, 2022 
    """
    # Initialize vector to store the flag. 1 means valid sample for training
    flag = np.ones(x.shape[0])

    for i in prange(x.shape[0]):  # iterate through all samples

        # too early, not enough information
        if i < seq_length - 1:
            flag[i] = 0  
            continue

        if check_NaN:
            # any NaN in the dynamic inputs makes the sample invalid
            x_sample = x[i-seq_length+1 : i+1, :]
            if np.any(np.isnan(x_sample)):
                flag[i] = 0
                continue

        if check_NaN:
            # all-NaN in the targets makes the sample invalid
            y_sample = y[i-predict_last_n+1 : i+1]
            if np.all(np.isnan(y_sample)):
                flag[i] = 0
                continue

        # any NaN in the static features makes the sample invalid
        if attributes is not None and check_NaN:
            if np.any(np.isnan(attributes)):
                flag[i] = 0

    return flag