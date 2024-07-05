# Calibration methods based on the library [Spotpy](https://doi.org/10.1371/journal.pone.0145180)[1]. 
# 
# **References:**
# [1]: "Houska, T., Kraft, P., Chamorro-Chavez, A. and Breuer, L.: SPOTting Model Parameters Using a Ready-Made Python Package, PLoS ONE, 10(12), e0145180, doi:10.1371/journal.pone.0145180, 2015"

#Import necessary packages
import numpy as np
import spotpy


class sce():
    """ Shuffled Complex Evolution Algorithm (SCE-UA)

    The actual calibration method is done internally by the Spotpy library[#]_, here we just create a wrapper to define
    the parameters and run the optimization . For a more detailed explanation of how the parameters are used, we refer
    to the Spotpy library.
 
    Parameters
    ----------
    name: str
        name of the algorithm
    repetitions: int
        maximum number of function evaluations allowed during optimization
    ngs: int
        number of complexes (sub-populations), take more than the number of
        analysed parameters
    kstop: int
        the number of past evolution loops and their respective objective value to assess whether the marginal improvement at the current loop (in percentage) is less than pcento
    pcento: float
        the percentage change allowed in the past kstop loops below which convergence is assumed to be achieved.
    peps: float
        Value of the normalized geometric range of the parameters in the population below which convergence is deemed achieved.
    max_loop_inc: int
        Number of loops executed at max in this function call
    random_state: int
        To have reproducible results
    References
    ----------
    .. [#] "Houska, T., Kraft, P., Chamorro-Chavez, A. and Breuer, L.: SPOTting Model Parameters Using a Ready-Made 
    Python Package, PLoS ONE, 10(12), e0145180, doi:10.1371/journal.pone.0145180, 2015"
    """
    # the default values are based on the example provided at: https://spotpy.readthedocs.io/en/latest/Calibration_with_SCE-UA/
    def __init__(self, 
                 name:str = 'sce',
                 repetitions: int = 5_000,
                 ngs: int = 7,
                 kstop:int = 100,
                 peps: float = 0.1,
                 pcento: float =0.1,
                 max_loop_inc: int=None,
                 random_state: int = None
                 ):
        
        self.name = name
        self.repetitions = repetitions
        self.ngs = ngs
        self.kstop = kstop
        self.peps = peps
        self.pcento = pcento
        self.max_loop_inc = max_loop_inc
        self.random_state = random_state

    def run_calibration(self, calibration_obj, path_output:str):
        """ Run calibration method
        
        Parameters
        ----------
        calibration_obj:
            Object from the class calibration_object
        path_output: str
            path where the calibration files will be stored
        file_id: str
            id of the object that is being calibrated (e.g basin_id)
        """

        file_name = path_output + calibration_obj.model.name + '_'  + self.name + '_' + calibration_obj.basin_id
        
        sampler=spotpy.algorithms.sceua(calibration_obj, dbname=file_name, dbformat='csv', save_sim=False,
                                        random_state = self.random_state)
        sampler.sample(repetitions=self.repetitions, 
                       ngs=self.ngs, 
                       kstop=self.kstop, 
                       peps=self.peps, 
                       pcento=self.pcento,
                       max_loop_inc=self.max_loop_inc
                       )
        
        return sampler
        

class dream():
    """ DiffeRential Evolution Adaptive Metropolis (DREAM) algorithhm

    The actual calibration method is done internally by the Spotpy library[#]_, here we just create a wrapper to define
    the parameters and run the optimization . For a more detailed explanation of how the parameters are used, we refer
    to the Spotpy library.

    For a detail explanation of the method see:
    Vrugt, J. A. (2016) Markov chain Monte Carlo simulation using the DREAM software package.
    
    Parameters
    ----------

    References
    ----------
    .. [#] "Houska, T., Kraft, P., Chamorro-Chavez, A. and Breuer, L.: SPOTting Model Parameters Using a Ready-Made 
    Python Package, PLoS ONE, 10(12), e0145180, doi:10.1371/journal.pone.0145180, 2015"
    """
    
    def __init__(self, 
                 name:str = 'dream',
                 repetitions: int = 5_000,
                 nChains: int = 7,
                 nCr: int = 3,
                 delta: int =3,
                 c: float = 0.1,
                 eps: float = 10e-6,
                 convergence_limit: float=1.2,
                 runs_after_convergence: int=100,
                 acceptance_test_option: int=6,
                 random_state: int = None
                 ):
          
        self.name = name
        self.repetitions = repetitions
        self.nChains = nChains
        self.nCr=nCr
        self.delta=delta
        self.c = c
        self.eps = eps
        self.convergence_limit = convergence_limit
        self.runs_after_convergence = runs_after_convergence
        self.acceptance_test_option = acceptance_test_option
        self.random_state = random_state


    def run_calibration(self, calibration_obj, path_output:str):
        """ Run calibration method
        
        Parameters
        ----------
        calibration_obj:
            Object from the class calibration_object
        path_output: str
            path where the calibration files will be stored
        file_id: str
            id of the object that is being calibrated (e.g basin_id)
        """

        file_name = path_output + calibration_obj.model.name + '_'  + self.name + '_' + calibration_obj.basin_id

        sampler = spotpy.algorithms.dream(calibration_obj, dbname=file_name, dbformat="csv", save_sim=False, 
                                          random_state = self.random_state)
        sampler = sampler.sample(repetitions=self.repetitions,
                                 nChains=self.nChains,
                                 nCr=self.nCr,
                                 delta=self.delta,
                                 c=self.c,
                                 eps=self.eps,
                                 convergence_limit=self.convergence_limit,
                                 runs_after_convergence=self.runs_after_convergence,
                                 acceptance_test_option=self.acceptance_test_option)
              
        return sampler


class rope():
    """ Robust Parameter Estimation (ROPE) algorithm

    The actual calibration method is done internally by the Spotpy library[#]_, here we just create a wrapper to define
    the parameters and run the optimization . For a more detailed explanation of how the parameters are used, we refer
    to the Spotpy library.

    For a detail explanation of the method see:
    Bárdossy, A. and Singh, S. K.: Robust estimation of hydrological model parameters,  Hydrol. Earth Syst. Sci. Discuss., 5(3), 1641–1675, 2008.
    
    Parameters
    ----------
    name: str
        name of the algorithm
    repetitions: int
        maximum number of function evaluations allowed during optimization
    repetitions_first_run: int
        Number of runs in the first rune
    repetitions_following_runs: int
        Number of runs for all following runs
    subsets: int 
        Number of time the rope algorithm creates a smaller searchwindows for parameters
    percentage_first_run: float
        Amount of runs that will be used for the next step after the first subset
    percentage_following_runs: float
        Amount of runs that will be used for the next step after in all following subsets
    NDIR: int
        Number of samples to draw


    References
    ----------
    .. [#] "Houska, T., Kraft, P., Chamorro-Chavez, A. and Breuer, L.: SPOTting Model Parameters Using a Ready-Made 
    Python Package, PLoS ONE, 10(12), e0145180, doi:10.1371/journal.pone.0145180, 2015"
    """
    
    def __init__(self, 
                 name:str = 'rope',
                 repetitions: int = 5_000,
                 repetitions_first_run: int=None,
                 subsets: int = 3,
                 percentage_first_run: float =0.1,
                 percentage_following_runs: float = 0.1,
                 NDIR: int = None,
                 random_state: int = None
                 ):
            
        
        self.name = name
        self.repetitions = repetitions
        self.repetitions_first_run = repetitions_first_run
        self.subsets = subsets
        self.percentage_first_run=percentage_first_run
        self.percentage_following_runs=percentage_following_runs
        self.NDIR = NDIR
        self.random_state = random_state


    def run_calibration(self, calibration_obj, path_output:str):
        """ Run calibration method
        
        Parameters
        ----------
        calibration_obj:
            Object from the class calibration_object
        path_output: str
            path where the calibration files will be stored
        file_id: str
            id of the object that is being calibrated (e.g basin_id)
        """

        file_name = path_output + calibration_obj.model.name + '_'  + self.name + '_' + calibration_obj.basin_id

        sampler = spotpy.algorithms.rope(calibration_obj, dbname=file_name, dbformat="csv", save_sim=False, 
                                         random_state = self.random_state)
        sampler = sampler.sample(repetitions=self.repetitions,
                                 repetitions_first_run = self.repetitions_first_run,
                                 subsets = self.subsets,
                                 percentage_first_run=self.percentage_first_run,
                                 percentage_following_runs=self.percentage_following_runs,
                                 NDIR = self.NDIR)
              
        return sampler