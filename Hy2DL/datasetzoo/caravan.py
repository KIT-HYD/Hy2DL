import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
from basedataset import BaseDataset



# Class that contain the methods used to read the information from Caravan dataset
class CARAVAN(BaseDataset):
    """Class to process the Caravans data set by [1]_ . 
    
    The class inherits from BaseDataset to execute the operations on how to load and process the data. However here we
    code the _read_attributes and _read_data methods, that specify how we should read the information from Caravans.
    The code would also run with user created datasets which conform to the Caravan style convention.

    This class and its methods were taken from Neural Hydrology [2]_ and adapted for our specific case. 
        
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
    check_Nan: : Optional[bool] = True
        Boolean that indicate if one should check of NaN values while processing the data
    
    References
    ----------
    .. [1] Kratzert, F., Nearing, G., Addor, N., Erickson, T., Gauch, M., Gilon, O., 
       Gudmundsson, L., Hassidim, A., Klotz, D., Nevo, S., Shalev, G., Matias, Y., 2023. Caravan - A global community dataset for 
       large-sample hydrology. Sci. Data 10, 61. https://doi.org/10.1038/s41597-023-01975-w
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
        super(CARAVAN, self).__init__(dynamic_input = dynamic_input,
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
        """Read the catchments` attributes from Caravan

        Parameters
        ----------
        path_data : Path to the root directory of Caravan that has to include a sub-directory 
                    called 'attributes' which contain the attributes of all sub-datasets in separate folders.

        Returns
        -------
        df : pd.DataFrame
            Dataframe with the catchments` attributes
        """
        data_dir = Path(self.path_data)
        # Take care of the subset directories in Caravans
        subdataset_dirs = [d for d in (data_dir / "attributes").glob('*') if d.is_dir()]

        # Load all required attribute files.
        dfs = []

        for subdataset_dir in subdataset_dirs: # Loop over each sub directory
            dfr_list = []
            for csv_file in subdataset_dir.glob("*.csv"): # Loop over each csv file
                dfr_list.append(pd.read_csv(csv_file, index_col="gauge_id"))
            dfr = pd.concat(dfr_list, axis=1)        
            dfs.append(dfr)

        # Merge all DataFrames along the basin index.
        df_attributes = pd.concat(dfs, axis=0)   

         # Encode categorical attributes in case there are any
        for column in df_attributes.columns:
            if df_attributes[column].dtype not in ['float64', 'int64']:
                df_attributes[column], _ = pd.factorize(df_attributes[column], sort=True) 
                

         # Filter attributes and basins of interest
        df_attributes = df_attributes.loc[self.entities_ids, self.static_input]

        return df_attributes
    
    def _read_data(self, catch_id: str)-> pd.DataFrame:
        """Loads the timeseries data of one basin from the Caravan dataset.
    
        Parameters
        ----------
        data_dir : Path
            Path to the root directory of Caravan that has to include a sub-directory called 'timeseries'. This 
            sub-directory has to contain another sub-directory called 'csv'.
        basin : str
            The Caravan gauge id string in the form of {subdataset_name}_{gauge_id}.

        Returns
        -------
        df: pd.DataFrame
            Dataframe with the catchments` timeseries
        """
        data_dir = Path(self.path_data)
        basin = catch_id

        # Get the subdataset name from the basin string.
        subdataset_name = basin.split('_')[0].lower()
        filepath = data_dir / "timeseries" / "csv" / subdataset_name / f"{basin}.csv"
        df = pd.read_csv(filepath, parse_dates=['date'])
        df = df.set_index('date')

        # Add other variables if required - Comment out this section in main branch (bwcluster access)
        # Create a new column "qobs_lead" with discharge values shifted by 7 time units
        #df['qobs_lead'] = df['qobs'].shift(-7)

        return df    