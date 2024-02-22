import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit, prange
from typing import List, Tuple


class CAMELS_GB():
    """Class to process the CAMELS GB data set by [#]_ . 
    
    This class and its methods were taken from Neural Hydrology [#]_ and adapted for our specific case. 
        
    References
    ----------
    .. [#] Coxon, G., Addor, N., Bloomfield, J. P., Freer, J., Fry, M., Hannaford, J., Howden, N. J. K., Lane, R., 
        Lewis, M., Robinson, E. L., Wagener, T., and Woods, R.: CAMELS-GB: Hydrometeorological time series and landscape 
        attributes for 671 catchments in Great Britain, Earth Syst. Sci. Data Discuss., 
        https://doi.org/10.5194/essd-2020-49, in review, 2020.
    .. [#] F. Kratzert, M. Gauch, G. Nearing and D. Klotz: NeuralHydrology -- A Python library for Deep Learning
        research in hydrology. Journal of Open Source Software, 7, 4050, doi: 10.21105/joss.04050, 2022 
    """
    
    @staticmethod
    def read_attributes(path_data: str) -> pd.DataFrame:
        """Read the catchments` attributes

        Parameters
        ----------
        path_data : str
            Path to the CAMELS GB directory.

        Returns
        -------
        df: pd.DataFrame
            Dataframe with the catchments` attributes
        """
        # files that contain the attributes
        path_attributes = Path(path_data)
        read_files = list(path_attributes.glob('*_attributes.csv'))

        dfs = []
        # Read each CSV file into a DataFrame and store it in list
        for file in read_files:
            df = pd.read_csv(file, sep=',', header=0, dtype={'gauge_id': str})
            df.set_index('gauge_id', inplace=True)
            dfs.append(df)
        # Join all dataframes
        df_attributes= pd.concat(dfs, axis=1)

        return df_attributes
    
    @staticmethod
    def read_data(path_data: str, catch_id: str, forcings: List[str]=None)-> pd.DataFrame:
        """Read the catchments` timeseries

        Parameters
        ----------
        path_data : str
            Path to the CAMELS GB directory.
        catch_id : str
            identifier of the basin.
        forcings : List[str]
            Not used, is just to have consistency with CAMELS-US. 

        Returns
        -------
        df: pd.DataFrame
            Dataframe with the catchments` timeseries
        """
        path_timeseries = Path(path_data) / 'timeseries' / f'CAMELS_GB_hydromet_timeseries_{catch_id}_19701001-20150930.csv'
        # load time series
        df = pd.read_csv(path_timeseries)
        df = df.set_index('date')
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
        return df

# Class that contain the methods used to read the information from CAMELS_US
class CAMELS_US():
    """Class to process the CAMELS US data set by [#]_ and [#]_. 
    
    This class and its methods were taken from Neural Hydrology [#]_ and adapted for our specific case. 
        
    References
    ----------
    .. [#] A. J. Newman, M. P. Clark, K. Sampson, A. Wood, L. E. Hay, A. Bock, R. J. Viger, D. Blodgett, 
        L. Brekke, J. R. Arnold, T. Hopson, and Q. Duan: Development of a large-sample watershed-scale 
        hydrometeorological dataset for the contiguous USA: dataset characteristics and assessment of regional 
        variability in hydrologic model performance. Hydrol. Earth Syst. Sci., 19, 209-223, 
        doi:10.5194/hess-19-209-2015, 2015
    .. [#] Addor, N., Newman, A. J., Mizukami, N. and Clark, M. P.: The CAMELS data set: catchment attributes and 
        meteorology for large-sample studies, Hydrol. Earth Syst. Sci., 21, 5293-5313, doi:10.5194/hess-21-5293-2017,
        2017.
    .. [#] F. Kratzert, M. Gauch, G. Nearing and D. Klotz: NeuralHydrology -- A Python library for Deep Learning
        research in hydrology. Journal of Open Source Software, 7, 4050, doi: 10.21105/joss.04050, 2022 
    """

    @staticmethod
    def read_attributes(path_data: str)-> pd.DataFrame:
        """Read the catchments` attributes

        Parameters
        ----------
        path_data : str
            Path to the CAMELS US directory.

        Returns
        -------
        df: pd.DataFrame
            Dataframe with the catchments` attributes
        """
        # files that contain the attributes
        path_attributes = Path(path_data) / 'camels_attributes_v2.0'
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

        return df
    
    @staticmethod
    def read_data(path_data: str, catch_id: str, forcings: List[str]) -> pd.DataFrame:
        """Read a specific catchment timeseries into a dataframe.

        Parameters
        ----------
        path_data : str
            Path to the CAMELS US directory.
        catch_id : str
            8-digit USGS identifier of the basin.
        forcings : str
            Can be e.g. 'daymet' or 'nldas', etc. Must match the folder names in the 'basin_mean_forcing' directory. 

        Returns
        -------
        df: pd.DataFrame
            Dataframe with the catchments` timeseries
        """
        # Read forcings 
        dfs = []
        for forcing in forcings: #forcings can be daymet, maurer or nldas
            df, area = CAMELS_US._load_forcing(path_data, catch_id, forcing)
            # rename columns in case there are multiple forcings
            if len(forcings) > 1:
                df = df.rename(columns={col: f"{col}_{forcing}" for col in df.columns})
            # Append to list
            dfs.append(df)

        df = pd.concat(dfs, axis=1) #dataframe with all the dynamic forcings

        # Read discharges and add them to current dataframe
        df['QObs(mm/d)'] = CAMELS_US._load_discharge(path_data, catch_id, area)

        # replace invalid discharge values by NaNs
        df['QObs(mm/d)'] = df['QObs(mm/d)'].apply(lambda x: np.nan if x < 0 else x)

        return df

    @staticmethod
    def _load_forcing(path_data: str, catch_id: str, forcing: str) -> Tuple[pd.DataFrame, int]:
        """Read a specific catchment forcing timeseries

        Parameters
        ----------
        path_data : str
            Path to the CAMELS US forcings` directory. This folder must contain a 'basin_mean_forcing' folder containing 
            one subdirectory for each forcing. The forcing directories have to contain 18 subdirectories 
            (for the 18 HUCS) as in the original CAMELS data set. In each HUC folder are the forcing files (.txt), 
            starting with the 8-digit basin id.
        catch_id : str
            8-digit USGS identifier of the basin.
        forcings : str
            Can be e.g. 'daymet' or 'nldas', etc. Must match the folder names in the 'basin_mean_forcing' directory. 

        Returns
        -------
        df: pd.DataFrame
            Dataframe with the catchments` timeseries
        area: int
            Catchment area (m2), specified in the header of the forcing file.
        """
    
        # Create a path to read the data
        forcing_path = Path(path_data) / 'basin_mean_forcing' / forcing
        file_path = list(forcing_path.glob(f'**/{catch_id}_*_forcing_leap.txt'))
        file_path = file_path[0]
        # Read dataframe
        with open(file_path, 'r') as fp:
            # load area from header
            fp.readline()
            fp.readline()
            area = int(fp.readline())
            # load the dataframe from the rest of the stream
            df = pd.read_csv(fp, sep='\s+')
            df["date"] = pd.to_datetime(df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str),
                                        format="%Y/%m/%d")
            
            df = df.set_index("date")

        return df, area
    
    @staticmethod
    def _load_discharge(path_data: str, catch_id: str, area: int)-> pd.DataFrame:
        """Read a specific catchment discharge timeseries

        Parameters
        ----------
        path_data : str
            Path to the CAMELS US directory. This folder must contain a 'usgs_streamflow' folder with 18
            subdirectories (for the 18 HUCS) as in the original CAMELS data set. In each HUC folder are the discharge 
            files (.txt), starting with the 8-digit basin id.
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
        streamflow_path = Path(path_data) / 'usgs_streamflow'
        file_path = list(streamflow_path.glob(f'**/{catch_id}_streamflow_qc.txt'))
        file_path = file_path[0]
        
        col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
        df = pd.read_csv(file_path, sep=r'\s+', header=None, names=col_names)
        df["date"] = pd.to_datetime(df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str), format="%Y/%m/%d")
        df = df.set_index("date")

        # normalize discharge from cubic feet per second to mm per day
        df.QObs = 28316846.592 * df.QObs * 86400 / (area * 10**6)

        return df.QObs


@njit()
def validate_samples(x: np.ndarray, y: np.ndarray, attributes: np.ndarray, seq_length: int, check_NaN:bool=True, 
                     predict_last_n:int=1) -> np.ndarray:
    
    """Checks for invalid samples due to NaN or insufficient sequence length.

    This function was taken from Neural Hydrology [#]_ and adapted for our specific case. 
        
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
    check_NaN : bool
        Boolean to specify if Nan should be checked or not
    predict_last_n: int
        Number of values that want to be used to calculate the loss

    Returns
    -------
    flag:np.ndarray 
        Array has a value of 1 for valid samples and a value of 0 for invalid samples.

    References
    ----------
    .. [#] F. Kratzert, M. Gauch, G. Nearing and D. Klotz: NeuralHydrology -- A Python library for Deep Learning
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