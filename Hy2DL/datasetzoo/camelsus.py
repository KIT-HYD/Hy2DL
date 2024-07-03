# import necessary packages
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from basedataset import BaseDataset


class CAMELS_US(BaseDataset):
    """Class to process the CAMELS US data set by [1]_ and [2]_. 
    
    The class inherits from BaseDataset to execute the operations on how to load and process the data. However here we
    code the _read_attributes and _read_data methods, that specify how we should read the information from CAMELS-US.

    This class and its methods were taken from Neural Hydrology [3]_ and adapted for our specific case. 
        
    Parameters
    ----------
    dynamic_input : List[str]
        name of variables used as dynamic series input in the lstm
    forcing : List[str]
        specificy which forcing data will be used (e.g. daymet, maurer, ndlas, etc.)
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
    conceptual_input: Optional[List[str]] = []
        Optional parameter. We need this when we use hybrid models. Name of variables used as dynamic series input in 
        the conceptual model
    check_Nan : : Optional[bool] = True
        Boolean that indicate if one should check of NaN values while processing the data
    
    References
    ----------
    .. [1] A. J. Newman, M. P. Clark, K. Sampson, A. Wood, L. E. Hay, A. Bock, R. J. Viger, D. Blodgett, 
        L. Brekke, J. R. Arnold, T. Hopson, and Q. Duan: Development of a large-sample watershed-scale 
        hydrometeorological dataset for the contiguous USA: dataset characteristics and assessment of regional 
        variability in hydrologic model performance. Hydrol. Earth Syst. Sci., 19, 209-223, 
        doi:10.5194/hess-19-209-2015, 2015
    .. [2] Addor, N., Newman, A. J., Mizukami, N. and Clark, M. P.: The CAMELS data set: catchment attributes and 
        meteorology for large-sample studies, Hydrol. Earth Syst. Sci., 21, 5293-5313, doi:10.5194/hess-21-5293-2017,
        2017.
    .. [3] F. Kratzert, M. Gauch, G. Nearing and D. Klotz: NeuralHydrology -- A Python library for Deep Learning
        research in hydrology. Journal of Open Source Software, 7, 4050, doi: 10.21105/joss.04050, 2022 
    """
    def __init__(self, 
                 dynamic_input: List[str],
                 forcing: List[str],
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
        self.forcing =  forcing
        super(CAMELS_US, self).__init__(dynamic_input = dynamic_input,
                                        target = target, 
                                        sequence_length = sequence_length,
                                        time_period = time_period,
                                        path_data = path_data,
                                        path_entities = path_entities,
                                        entity= entity,
                                        path_additional_features = path_additional_features,
                                        predict_last_n = predict_last_n,
                                        static_input = static_input,
                                        conceptual_input = conceptual_input,
                                        check_NaN=check_NaN)
    

    def _read_attributes(self)-> pd.DataFrame:
        """Read the catchments` attributes

        Returns
        -------
        df : pd.DataFrame
            Dataframe with the catchments` attributes
        """
        # files that contain the attributes
        path_attributes = Path(self.path_data) / 'camels_attributes_v2.0'
        read_files = list(path_attributes.glob('camels_*.txt'))

        # Read one by one the attributes files
        dfs = []
        for file in read_files:
            df_temp = pd.read_csv(file, sep=';', header=0, dtype={'gauge_id': str})
            df_temp = df_temp.set_index('gauge_id')
            dfs.append(df_temp)

        # Concatenate all the dataframes into a single one
        df = pd.concat(dfs, axis=1)
        # convert huc column to double digit strings
        df['huc'] = df['huc_02'].apply(lambda x: str(x).zfill(2))
        df = df.drop('huc_02', axis=1)

        # Encode categorical attributes in case there are any
        for column in df.columns:
            if df[column].dtype not in ['float64', 'int64']:
                df[column], _ = pd.factorize(df[column], sort=True)

        # Filter attributes and basins of interest
        df = df.loc[self.entities_ids, self.static_input]
        
        return df
    
    def _read_data(self, catch_id: str) -> pd.DataFrame:
        """Read a specific catchment timeseries into a dataframe.

        Parameters
        ----------
        catch_id : str
            8-digit USGS identifier of the basin.

        Returns
        -------
        df : pd.DataFrame
            Dataframe with the catchments` timeseries
        """
        # Read forcings 
        dfs = []
        for forcing in self.forcing: #forcings can be daymet, maurer or nldas
            df, area = self._load_forcing(catch_id=catch_id, forcing=forcing)
            # rename columns in case there are multiple forcings
            if len(self.forcing) > 1:
                df = df.rename(columns={col: f"{col}_{forcing}" for col in df.columns})
            # Append to list
            dfs.append(df)

        df = pd.concat(dfs, axis=1) #dataframe with all the dynamic forcings

        # Read discharges and add them to current dataframe
        df['QObs(mm/d)'] = self._load_discharge(catch_id=catch_id, area=area)

        # replace invalid discharge values by NaNs
        df['QObs(mm/d)'] = df['QObs(mm/d)'].apply(lambda x: np.nan if x < 0 else x)

        return df

    def _load_forcing(self, catch_id: str, forcing : str) -> Tuple[pd.DataFrame, int]:
        """Read a specific catchment forcing timeseries

        Parameters
        ----------
        catch_id : str
            8-digit USGS identifier of the basin.
        forcing : str
            Can be e.g. 'daymet' or 'nldas', etc. Must match the folder names in the 'basin_mean_forcing' directory. 

        Returns
        -------
        df : pd.DataFrame
            Dataframe with the catchments` timeseries
        area : int
            Catchment area (m2), specified in the header of the forcing file.
        """
        # Create a path to read the data
        forcing_path = Path(self.path_data) / 'basin_mean_forcing' / forcing
        file_path = list(forcing_path.glob(f'**/{catch_id}_*_forcing_leap.txt'))
        file_path = file_path[0]
        # Read dataframe
        with open(file_path, 'r') as fp:
            # load area from header
            fp.readline()
            fp.readline()
            area = int(fp.readline())
            # load the dataframe from the rest of the stream
            df = pd.read_csv(fp, sep=r'\s+')
            df["date"] = pd.to_datetime(df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str),
                                        format="%Y/%m/%d")
            
            df = df.set_index("date")

        return df, area
    
    def _load_discharge(self, catch_id: str, area: int)-> pd.DataFrame:
        """Read a specific catchment discharge timeseries

        Parameters
        ----------
        catch_id : str
            8-digit USGS identifier of the basin.
        area : int
            Catchment area (m2), used to normalize the discharge.

        Returns
        -------
        df: pd.Series
            Time-index pandas.Series of the discharge values (mm/day)
        """
        # Create a path to read the data
        streamflow_path = Path(self.path_data) / 'usgs_streamflow'
        file_path = list(streamflow_path.glob(f'**/{catch_id}_streamflow_qc.txt'))
        file_path = file_path[0]
        
        col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
        df = pd.read_csv(file_path, sep=r'\s+', header=None, names=col_names)
        df["date"] = pd.to_datetime(df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str), format="%Y/%m/%d")
        df = df.set_index("date")

        # normalize discharge from cubic feet per second to mm per day
        df.QObs = 28316846.592 * df.QObs * 86400 / (area * 10**6)

        return df.QObs