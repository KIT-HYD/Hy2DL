#Import necessary packages
import numpy as np
from typing import List, Dict, Tuple

class BaseConceptualModel():
    """Abstract base model class, don't use this class for model training!

    The purpose is to have some common operations that all conceptual models will need. 
    """

    def __init__(self,):
        super().__init__()


    def run_model(self, input: np.ndarray, param: List[float]) -> Tuple:
        raise NotImplementedError


    def _initialize_information(self, conceptual_inputs: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Initialize the structures to store the time evolution of the internal states and the outflow of the conceptual
        model

        Parameters
        ----------
        conceptual_inputs: np.ndarray
            Inputs of the conceptual model (dynamic forcings)

        Returns
        -------
        Tuple[np.ndarray, Dict[str, np.ndarray]]
            - q_out: np.ndarray
                Array to store the outputs of the conceptual model
            - states: Dict[str, np.ndarray]
                Dictionary to store the time evolution of the internal states (buckets) of the conceptual model
        """

        states = {}
        # initialize dictionary to store the evolution of the states
        for name, _ in self._initial_states.items():
            states[name] = np.zeros((conceptual_inputs.shape[0], 1))

        # initialize vectors to store the evolution of the outputs
        out = np.zeros((conceptual_inputs.shape[0], 1))

        return out, states
    
    @property
    def _initial_states(self)-> Dict[str, float]:
        raise NotImplementedError

    @property
    def _parameter_ranges(self) -> Dict[str, List[float]]:
        raise NotImplementedError


class SHM(BaseConceptualModel):
    """Slightly modified version of the SHM model [#]_.
    
    References
    ----------
    .. [#] Ehret, U., van Pruijssen, R., Bortoli, M., Loritz, R.,  Azmi, E. and Zehe, E: Adaptive clustering: reducing 
        the computational costs of distributed (hydrological) modelling by exploiting time-variable similarity among 
        model elements. HESS, 24, 4389-4411, doi: 10.5194/hess-24-4389-2020, 2020
    """
    
    def __init__(self):
        super(SHM, self).__init__()
        self.name = 'SHM'
    
    def run_model(self, input: np.ndarray, param: List[float]) -> Tuple:
        """Run the model
        
        Parameters
        ----------
        input : np.ndarray
            Inputs for the conceptual model
        param : List[float]
            Parameters of the model
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - out: np.ndarray
                outputs of the conceptual model
            - states: np.ndarray
                time evolution of the internal states (buckets) of the conceptual model   
        """
        # initialize structures to store the information
        out, states = self._initialize_information(conceptual_inputs=input)
        
        # read parameters
        dd, f_thr, sumax, beta, perc, kf, ki, kb = param 
        
        # Storages
        ss =  self._initial_states['ss']
        sf =  self._initial_states['sf']
        su =  self._initial_states['su']
        si =  self._initial_states['si']
        sb =  self._initial_states['sb']
        
        # run model for each timestep
        for i, (p, pet, temp) in enumerate(input):
            # Snow module --------------------------
            if temp > 0: # if there is snowmelt
                qs_out = min(ss, dd*temp) # snowmelt from snow reservoir
                ss = ss - qs_out # substract snowmelt from snow reservoir
                qsp_out = qs_out + p # flow from snowmelt and rainfall
            else: # if the is no snowmelt
                ss=ss + p # precipitation accumalates as snow in snow reservoir
                qsp_out = 0.0

            # Split snowmelt+rainfall into inflow to fastflow reservoir and unsaturated reservoir ------
            qf_in = max(0, qsp_out-f_thr)
            qu_in = min(qsp_out, f_thr)

            # Fastflow module ----------------------
            sf = sf + qf_in
            qf_out = sf/ kf
            sf = sf - qf_out

            # Unsaturated zone----------------------
            psi = (su / sumax) ** beta #[-]
            su_temp = su + qu_in * (1 - psi)
            su = min(su_temp, sumax)
            qu_out = qu_in * psi + max(0.0, su_temp - sumax) # [mm]
            
            # Evapotranspiration -------------------
            klu = 0.9 # land use correction factor [-]
            if su <= 0.0:
                ktetha = 0.0
            elif su >= 0.8 * sumax:
                ktetha = 1.0
            else:
                ktetha = su / sumax

            ret = pet * klu * ktetha #[mm]
            su = max(0.0, su - ret) #[mm]

            # Interflow reservoir ------------------
            qi_in = qu_out * perc #[mm]
            si = si + qi_in #[mm]
            qi_out = si / ki #[mm]
            si = si - qi_out #[mm]

            # Baseflow reservoir -------------------
            qb_in = qu_out * (1 - perc) #[mm]
            sb = sb + qb_in #[mm]
            qb_out = sb / kb #[mm]
            sb = sb - qb_out #[mm]

            # Store time evolution of the internal states
            states['ss'][i] = ss
            states['sf'][i] = sf
            states['su'][i] = su
            states['si'][i] = si
            states['sb'][i] = sb

            # total outflow
            out[i] = qf_out + qi_out + qb_out  # [mm]

        return out, states
 
    @property
    def _initial_states(self) -> Dict[str, float]:
        return {
            'ss'  : 0.0,
            'sf'  : 1.0,
            'su'  : 5.0,
            'si'  : 10.0,
            'sb'  : 15.0
            }
    
    @property
    def parameter_ranges(self) -> Dict[str, List[float]]:
        return {
            'dd': [0.0, 10.0],
            'f_thr'  : [10.0,60.0],
            'sumax'  : [20.0,700.0],
            'beta'  : [1.0, 6.0],
            'perc'  : [0.0, 1.0],
            'kf'  : [1.0, 20.0],
            'ki'  : [1.0, 100.0],
            'kb'  : [10.0, 1000.0]
            }
    

class bucket(BaseConceptualModel):
    """Model with a single linear reservoir
    """
    
    def __init__(self):
        super(bucket, self).__init__()
        self.name = 'bucket'
    
    def run_model(self, input: np.ndarray, param: List[float]) -> Tuple:
        """Run the model
        
        Parameters
        ----------
        input : np.ndarray
            Inputs for the conceptual model
        param : List[float]
            Parameters of the model
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - out: np.ndarray
                outputs of the conceptual model
            - states: np.ndarray
                time evolution of the internal states (buckets) of the conceptual model   
        """
        # initialize structures to store the information
        out, states = self._initialize_information(conceptual_inputs=input)
        # read parameters
        aux_ET, ki = param 
        # Storages
        si =  self._initial_states['si']
        
        # run model for each timestep
        for i, (p, pet, _) in enumerate(input):
            # 1 bucket reservoir ------------------
            si = si + p #[mm]
            ret = pet * aux_ET #[mm]
            si = max(0.0, si - ret) #[mm]
            qi_out = si / ki #[mm]
            si = si - qi_out #[mm]
            # Store time evolution of the internal states
            states['si'][i] = si
            # total outflow
            out[i] = qi_out  # [mm]

        return out, states
 
    @property
    def _initial_states(self) -> Dict[str, float]:
        return {
            'si'  : 5.0
            }
    
    @property
    def parameter_ranges(self) -> Dict[str, List[float]]:
        return {
            'aux_ET': [0.0, 1.5],
            'ki'  : [1.0,500.0]
            }


class NonSense(BaseConceptualModel):
    """Hydrological model with unfeasible structure.
    """
    
    def __init__(self):
        super(NonSense, self).__init__()
        self.name = 'NonSense'
    
    def run_model(self, input: np.ndarray, param: List[float]) -> Tuple:
        """Run the model
        
        Parameters
        ----------
        input : np.ndarray
            Inputs for the conceptual model
        param : List[float]
            Parameters of the model
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - out: np.ndarray
                outputs of the conceptual model
            - states: np.ndarray
                time evolution of the internal states (buckets) of the conceptual model   
        """
        # initialize structures to store the information
        out, states = self._initialize_information(conceptual_inputs=input)
        
        # read parameters
        dd, sumax, beta, ki, kb = param 
        
        # Storages
        ss =  self._initial_states['ss']
        su =  self._initial_states['su']
        si =  self._initial_states['si']
        sb =  self._initial_states['sb']
        
            # run model for each timestep
        for i, (p, pet, temp) in enumerate(input):
            
            # Snow module --------------------------
            if temp > 0: # if there is snowmelt
                qs_out = min(ss, dd*temp) # snowmelt from snow reservoir
                ss = ss - qs_out # substract snowmelt from snow reservoir
                qsp_out = qs_out + p # flow from snowmelt and rainfall
            else: # if the is no snowmelt
                ss=ss + p # precipitation accumalates as snow in snow reservoir
                qsp_out = 0.0

            # Baseflow reservoir -------------------
            sb = sb + qsp_out #[mm]
            qb_out = sb / kb #[mm]
            sb = sb - qb_out #[mm]

            # Interflow reservoir ------------------
            si = si + qb_out #[mm]
            qi_out = si / ki #[mm]
            si = si - qi_out #[mm]
            
            # Unsaturated zone----------------------
            psi = (su / sumax) ** beta #[-]
            su_temp = su + qi_out * (1 - psi)
            su = min(su_temp, sumax)
            qu_out = qi_out * psi + max(0.0, su_temp - sumax) # [mm]

            # Evapotranspiration -------------------
            klu = 0.9 # land use correction factor [-]
            if su <= 0.0:
                ktetha = 0.0
            elif su >= 0.8 * sumax:
                ktetha = 1.0
            else:
                ktetha = su / sumax

            ret = pet * klu * ktetha #[mm]
            su = max(0.0, su - ret) #[mm]
        
            # Store time evolution of the internal states
            states['ss'][i] = ss
            states['su'][i] = su
            states['si'][i] = si
            states['sb'][i] = sb

            # total outflow
            out[i] = qu_out  # [mm]

        return out, states
 
    @property
    def _initial_states(self) -> Dict[str, float]:
        return {
            'ss'  : 0.0,
            'su'  : 5.0,
            'si'  : 10.0,
            'sb'  : 15.0
            }
    
    @property
    def parameter_ranges(self) -> Dict[str, List[float]]:
        return {
            'dd': [0.0, 10.0],
            'sumax'  : [20.0,700.0],
            'beta'  : [1.0, 6.0],
            'ki'  : [1.0, 100.0],
            'kb'  : [10.0, 1000.0]
            }