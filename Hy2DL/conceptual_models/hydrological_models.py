#Import necessary packages
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import scipy

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
            'dd': (0.0, 10.0),
            'f_thr'  : (10.0,60.0),
            'sumax'  : (20.0,700.0),
            'beta'  : (1.0, 6.0),
            'perc'  : (0.0, 1.0),
            'kf'  : (1.0, 20.0),
            'ki'  : (1.0, 100.0),
            'kb'  : (10.0, 1000.0)
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
            'aux_ET': (0.0, 1.5),
            'ki'  : (1.0,500.0)
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
            'dd': (0.0, 10.0),
            'sumax'  : (20.0,700.0),
            'beta'  : (1.0, 6.0),
            'ki'  : (1.0, 100.0),
            'kb'  : (10.0, 1000.0)
            }
    

class HBV(BaseConceptualModel):
    """HBV model. 
    
    Implementation based on Feng et al. [1]_ and Seibert [2]_.

    References
    ----------
    .. [1] Feng, D., Liu, J., Lawson, K., & Shen, C. (2022). Differentiable, learnable, regionalized process-based 
        models with multiphysical outputs can approach state-of-the-art hydrologic prediction accuracy. Water Resources 
        Research, 58, e2022WR032404. https://doi.org/10.1029/2022WR032404
    .. [2] Seibert, J. (2005) HBV Light Version 2. User’s Manual. Department of Physical Geography and Quaternary 
        Geology, Stockholm University, Stockholm
    """
    def __init__(self):
        super(HBV, self).__init__()
        self.name = 'HBV'
    
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
        BETA, FC, K0, K1, K2, LP, PERC, UZL, TT, CFMAX, CFR, CWH, alpha, beta  = param 
        
        # Storages
        SNOWPACK =  self._initial_states['SNOWPACK']
        MELTWATER =  self._initial_states['MELTWATER']
        SM =  self._initial_states['SM']
        SUZ =  self._initial_states['SUZ']
        SLZ =  self._initial_states['SLZ']
        
        # run model for each timestep
        for i, (p, pet, temp) in enumerate(input):

            liquid_p, snow = (p, 0) if temp > TT else (0, p)

            # Snow module -----------------------------------------------------------------------------------------
            SNOWPACK = SNOWPACK + snow
            melt = CFMAX * (temp - TT)
            melt = max(melt, 0.0)
            melt = min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = CFR* CFMAX * (TT -  temp)
            refreezing = max(refreezing, 0.0)
            refreezing = min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = MELTWATER - (CWH* SNOWPACK)
            tosoil = max(tosoil, 0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil and evaporation ---------------------------------------------------------------------------------
            soil_wetness = (SM / FC) ** BETA
            soil_wetness = min(max(soil_wetness, 0.0), 1.0)
            recharge = (liquid_p + tosoil) * soil_wetness

            SM = SM + liquid_p + tosoil - recharge
            excess = SM - FC
            excess = max(excess,0.0)
            SM = SM - excess
            evapfactor = SM / (LP * FC)
            evapfactor  = min(max(evapfactor, 0.0), 1.0)
            ETact = pet * evapfactor
            ETact = min(SM, ETact)
            SM = max(SM - ETact, 0.0)
            
            # Groundwater boxes -------------------------------------------------------------------------------------
            SUZ = SUZ + recharge + excess
            PERCact = min(SUZ, PERC)
            SUZ = SUZ - PERCact
            Q0 = K0 * max(SUZ-UZL, 0.0)
            SUZ = SUZ - Q0
            Q1 = K1 * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERCact
            Q2 = K2 * SLZ
            SLZ = SLZ - Q2       
            
            # Store time evolution of the internal states
            states['SNOWPACK'][i] = SNOWPACK
            states['MELTWATER'][i] = MELTWATER
            states['SM'][i] = SM
            states['SUZ'][i] = SUZ
            states['SLZ'][i] = SLZ
            
            # total outflow
            out[i] = Q0 + Q1 + Q2 # [mm]
        
        # routing method
        UH =  self._gamma_routing(alpha = alpha, beta = beta, uh_len=15)
        out = self._uh_conv(discharge = out, unit_hydrograph=UH).reshape((-1, 1))
        
        return out, states
    
    def _gamma_routing(self, alpha:float, beta:float, uh_len:int = 10):
        """Unit hydrograph based on gamma function.

        Parameters
        ----------
        alpha: float
            Shape parameter of the Gamma distribution.
        beta: float
            Scale parameter of the Gamma distribution.
        uh_len: int
            Number of timesteps the unitary hydrograph will have.

        Returns
        -------
        uh : torch.Tensor
            Unit hydrograph
        """
        x = np.arange(0.5, 0.5+uh_len, 1)
        coeff = 1 / (beta**alpha * np.exp(scipy.special.loggamma(alpha)))
        gamma_pdf = coeff * (x**(alpha - 1)) * np.exp(-x / beta)
        # Normalize data so the sum of the pdf equals 1
        uh = gamma_pdf/np.sum(gamma_pdf)
        return uh

        
    def _uh_conv(self, discharge: np.ndarray, unit_hydrograph: np.ndarray):
        """Unitary hydrograph routing.

        Parameters
        ----------
        discharge: 
            Discharge series
        unit_hydrograph: 
            Unit hydrograph

        Returns
        -------
        y: 
            Routed discharge

        """
        padding_size = unit_hydrograph.shape[0] - 1
        y = np.convolve(np.array(discharge).flatten(), unit_hydrograph, mode='full')
        return y[0:-padding_size]

    
    @property
    def _initial_states(self) -> Dict[str, float]:
        return {
            'SNOWPACK' : 0.001,
            'MELTWATER' : 0.001,
            'SM' : 0.001,
            'SUZ' : 0.001,
            'SLZ' : 0.001
            }

    @property
    def parameter_ranges(self) -> Dict[str, List[float]]:
        return {
            'BETA': (1.0, 6.0),
            'FC'  : (50.0, 1000.0),
            'K0'  : (0.05, 0.9),
            'K1'  : (0.01, 0.5),
            'K2'  : (0.001, 0.2),
            'LP'  : (0.2, 1.0),
            'PERC' : (0.0, 10.0),
            'UZL' : (0.0, 100.0),
            'TT'  : (-2.5 , 2.5),
            'CFMAX' : (0.5, 10.0),
            'CFR' : (0.0, 0.1),
            'CWH' : (0.0, 0.2),
            'alpha' : (0.0, 2.9),
            'beta' : (0.0, 6.5)
            }