import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit, prange
from typing import List, Tuple

# Class that contain the methods used to read the information from Caravan dataset
class CARAVAN():
    """Class to process the Caravans data set by [#]_ and [#]_. 
    The Code will only work with Caravan csv files. Support is not provided for netcdf files
    This class and its methods were taken from Neural Hydrology [#]_ and adapted for our specific case. 
        
    References
    ----------
    .. [#] Kratzert, F., Nearing, G., Addor, N., Erickson, T., Gauch, M., Gilon, O., 
       Gudmundsson, L., Hassidim, A., Klotz, D., Nevo, S., Shalev, G., Matias, Y., 2023. Caravan - A global community dataset for 
       large-sample hydrology. Sci. Data 10, 61. https://doi.org/10.1038/s41597-023-01975-w
    .. [#] F. Kratzert, M. Gauch, G. Nearing and D. Klotz: NeuralHydrology -- A Python library for Deep Learning
        research in hydrology. Journal of Open Source Software, 7, 4050, doi: 10.21105/joss.04050, 2022 
    """

    @staticmethod
    def read_attributes(path_data: str) -> pd.DataFrame:
        """Read the catchments` attributes from Caravan

        Parameters
        ----------
        path_data   Path to the root directory of Caravan that has to include a sub-directory 
                    called 'attributes' which contain the attributes of all sub-datasets in separate folders.

        Returns
        -------
        df: pd.DataFrame
            Dataframe with the catchments` attributes
        """
        data_dir = Path(path_data)
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
        df = pd.concat(dfs, axis=0)    

        return df
    
    @staticmethod
    def read_data(path_data: str, catch_id: str)-> pd.DataFrame:
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
        data_dir = Path(path_data)
        basin = catch_id

        # Get the subdataset name from the basin string.
        subdataset_name = basin.split('_')[0]
        filepath = data_dir / "timeseries" / "csv" / subdataset_name / f"{basin}.csv"
        df = pd.read_csv(filepath, parse_dates=['date'])
        df = df.set_index('date')

        # Add other variables if required - Comment out this section in main branch
        # Create a new column "qobs_lead" with discharge values shifted by 7 time units
        #df['qobs_lead'] = df['qobs'].shift(-7)

        return df    