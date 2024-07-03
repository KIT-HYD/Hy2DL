# import necessary packages
import pandas as pd
from pathlib import Path
from typing import List, Optional
from basedataset import BaseDataset


class CAMELS_CH(BaseDataset):
    """Class to process the CAMELS CH data set by [#]_ . 
    
    The class inherits from BaseDataset to execute the operations on how to load and process the data. However here we
    code the _read_attributes and _read_data methods, that specify how we should read the information from CAMELS-CH.

    This class and its methods were taken from Neural Hydrology [#]_ and adapted for our specific case. 

    The CAMELS CH data set provides both observed and simulated static attributes as well as time series
    This code reads the observed static attributes and time series
        
    Parameters
    ----------
    dynamic_input : List[str]
        name of variables used as dynamic series input in the lstm
    target: List[str]
        target variable(s)
    sequence_length: int
        sequence length used for the model
    time_period: List[str]
        initial and final date (e.g. ['1987-10-01','1999-09-30']) of the time period of interest 
    path_data: str
        path to the folder were the data is stored
    path_entities: str
        path to a txt file that contain the id of the entities (e.g. catchment`s ids) that will be analyzed
    entity: str
        id of the entities (e.g. catchment`s id) that will be analyzed. Alternative option to specifying a
        path_entities.
    path_addional features: Optional[str] = None
        Optional parameter. Allows the option to add any arbitrary data that is not included in the standard data sets.
        Path to a pickle file (or list of paths for multiple files), containing a dictionary with each key corresponding 
        to one basin id and the value is a date-time indexed pandas DataFrame.      
    predict_last_n: Optional[int] = 1
        number of timesteps (e.g. days) used to calculate the loss
    static_input : Optional[List[str]] = []
        name of static inputs used as input in the lstm (e.g. catchment attributes)
    conceptual_input: Optional[List[str]] = []
        Optional parameter. We need this when we use hybrid models. Name of variables used as dynamic series input in 
        the conceptual model
    check_Nan: : Optional[bool] = True
        Boolean that indicate if one should check of NaN values while processing the data
    
    References
    ----------
    .. [#] Höge, M., Kauzlaric, M., Siber, R., Schönenberger, U., Horton, P., Schwanbeck, J., Floriancic,
        M. G., Viviroli, D., Wilhelm, S., Sikorska-Senoner, A. E., Addor, N., Brunner, M., Pool, S., Zappa, M.,
        and Fenicia, F.: CAMELS-CH: hydro-meteorological time series and landscape attributes for 331 catchments
        in hydrologic Switzerland, Earth Syst. Sci. Data, 15, 5755–5784,
        https://doi.org/10.5194/essd-15-5755-2023, 2023.
    .. [#] F. Kratzert, M. Gauch, G. Nearing and D. Klotz: NeuralHydrology -- A Python library for Deep Learning
        research in hydrology. Journal of Open Source Software, 7, 4050, doi: 10.21105/joss.04050, 2022 
    """
    
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
        
        # Run the __init__ method of BaseDataset class, where the data is processed
        super(CAMELS_CH, self).__init__(dynamic_input = dynamic_input,
                                        target = target, 
                                        sequence_length = sequence_length,
                                        time_period = time_period,
                                        path_data = path_data,
                                        path_entities = path_entities,
                                        entity = entity,
                                        path_additional_features = path_additional_features,
                                        predict_last_n = predict_last_n,
                                        static_input = static_input,
                                        conceptual_input = conceptual_input,
                                        check_NaN=check_NaN)

    def _read_attributes(self) -> pd.DataFrame:
        """Read the catchments` attributes

        Returns
        -------
        df: pd.DataFrame
            Dataframe with the catchments` attributes
        """
        # files that contain the attributes
        path_attributes = Path(self.path_data) / 'static_attributes'
        read_files = list(path_attributes.glob('CAMELS_CH_*.csv'))

        dfs = []
        # Read each CSV file into a DataFrame and store it in list
        for file in read_files:
            df = pd.read_csv(file, sep=',', header=0, dtype={'gauge_id': str}, skiprows=1, encoding='iso-8859-1')
            df.set_index('gauge_id', inplace=True)
            dfs.append(df)
        
        # Join all dataframes
        df_attributes= pd.concat(dfs, axis=1)
        
        # Encode categorical attributes in case there are any
        for column in df_attributes.columns:
            if df_attributes[column].dtype not in ['float64', 'int64']:
                df_attributes[column], _ = pd.factorize(df_attributes[column], sort=True)
        
        # Replace nan by the mean value of the respective column
        #df_attributes = df_attributes.fillna(df_attributes.mean())
        
        # Filter attributes and basins of interest
        df_attributes = df_attributes.loc[self.entities_ids, self.static_input]

        return df_attributes

    
    def _read_data(self, catch_id: str)-> pd.DataFrame:
        """Read the catchments` timeseries

        Parameters
        ----------
        catch_id : str
            identifier of the basin.

        Returns
        -------
        df: pd.DataFrame
            Dataframe with the catchments` timeseries
        """
        path_timeseries_obs = Path(self.path_data) / 'timeseries' / 'observation_based' / f'CAMELS_CH_obs_based_{catch_id}.csv'
        # load time series
        df_obs = pd.read_csv(path_timeseries_obs)
        df_obs = df_obs.set_index('date')
        df_obs.index = pd.to_datetime(df_obs.index, format="%Y-%m-%d")

        # adding simulated time series
        path_timeseries_sim = Path(self.path_data) / 'timeseries' / 'simulation_based' / f'CAMELS_CH_sim_based_{catch_id}.csv'
        # load time series
        df_sim = pd.read_csv(path_timeseries_sim)
        df_sim = df_sim.set_index('date')
        df_sim.index = pd.to_datetime(df_sim.index, format="%Y-%m-%d")

        # concatenating observed timeseries with simulated time series
        df = pd.concat([df_obs, df_sim], axis=1)
        
        return df