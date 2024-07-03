# import necessary packages
import pandas as pd
from pathlib import Path
from typing import List, Optional
from basedataset import BaseDataset


class CAMELS_GB(BaseDataset):
    """Class to process the CAMELS GB data set by [1]_ . 
    
    The class inherits from BaseDataset to execute the operations on how to load and process the data. However here we
    code the _read_attributes and _read_data methods, that specify how we should read the information from CAMELS-GB.

    This class and its methods were taken from Neural Hydrology [2]_ and adapted for our specific case. 
        
    Parameters
    ----------
    dynamic_input : List[str]
        name of variables used as dynamic series input in the lstm
    target: List[str]
        target variable(s)
    sequence_length: int
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
    check_Nan: : Optional[bool] = True
        Boolean that indicate if one should check of NaN values while processing the data
    
    References
    ----------
    .. [1] Coxon, G., Addor, N., Bloomfield, J. P., Freer, J., Fry, M., Hannaford, J., Howden, N. J. K., Lane, R., 
        Lewis, M., Robinson, E. L., Wagener, T., and Woods, R.: CAMELS-GB: Hydrometeorological time series and landscape 
        attributes for 671 catchments in Great Britain, Earth Syst. Sci. Data Discuss., 
        https://doi.org/10.5194/essd-2020-49, in review, 2020.
    .. [2] F. Kratzert, M. Gauch, G. Nearing and D. Klotz: NeuralHydrology -- A Python library for Deep Learning
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
        super(CAMELS_GB, self).__init__(dynamic_input = dynamic_input,
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
        df : pd.DataFrame
            Dataframe with the catchments` attributes
        """
        # files that contain the attributes
        path_attributes = Path(self.path_data)
        read_files = list(path_attributes.glob('*_attributes.csv'))

        dfs = []
        # Read each CSV file into a DataFrame and store it in list
        for file in read_files:
            df = pd.read_csv(file, sep=',', header=0, dtype={'gauge_id': str})
            df.set_index('gauge_id', inplace=True)
            dfs.append(df)
        
        # Join all dataframes
        df_attributes= pd.concat(dfs, axis=1)
        
        # Encode categorical attributes in case there are any
        for column in df_attributes.columns:
            if df_attributes[column].dtype not in ['float64', 'int64']:
                df_attributes[column], _ = pd.factorize(df_attributes[column], sort=True)
        
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
        df : pd.DataFrame
            Dataframe with the catchments` timeseries
        """
        path_timeseries = Path(self.path_data) / 'timeseries' / f'CAMELS_GB_hydromet_timeseries_{catch_id}_19701001-20150930.csv'
        # load time series
        df = pd.read_csv(path_timeseries)
        df = df.set_index('date')
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
        return df