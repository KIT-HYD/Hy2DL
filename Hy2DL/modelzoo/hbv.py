from typing import List, Dict, Union, Optional, Tuple
import torch
from baseconceptualmodel import BaseConceptualModel

class HBV(BaseConceptualModel):
    """HBV model. 
    
    Implementation based on Feng et al. [1]_ and Seibert [2]_. The code creates a modified version of the HBV model that 
    can be used as a differentiable entity to create hybrid models. One can run multiple entities of the model at the 
    same time.

    Parameters
    ----------
    n_models : int
        Number of model entities that will be run at the same time
    parameter_type : List[str]
        List to specify which parameters of the conceptual model will be dynamic.  

    References
    ----------
    .. [1] Feng, D., Liu, J., Lawson, K., & Shen, C. (2022). Differentiable, learnable, regionalized process-based 
        models with multiphysical outputs can approach state-of-the-art hydrologic prediction accuracy. Water Resources 
        Research, 58, e2022WR032404. https://doi.org/10.1029/2022WR032404
    .. [2] Seibert, J. (2005) HBV Light Version 2. User’s Manual. Department of Physical Geography and Quaternary 
        Geology, Stockholm University, Stockholm
    """
    def __init__(self, n_models:int=1, parameter_type:List[str]=None):
        super(HBV, self).__init__()
        self.n_conceptual_models = n_models
        self.parameter_type = self._map_parameter_type(parameter_type=parameter_type)
        self.output_size = 1


    def forward(self, x_conceptual: torch.Tensor, parameters: Dict[str, torch.Tensor], 
                initial_states: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Union[torch.Tensor, 
                                                                                             Dict[str, torch.Tensor]]]:
        """Forward pass on the HBV model. 

        Parameters
        ----------
        x_conceptual: torch.Tensor
            Tensor of size [batch_size, time_steps, n_inputs]. The batch_size is associated with a certain basin and a
            certain prediction period. The time_steps refer to the number of time steps (e.g. days) that our conceptual
            model is going to be run for. The n_inputs refer to the dynamic forcings used to run the conceptual model
            (e.g. Precipitation, Temperature...)
        parameter_type : List[str]
            List to specify which parameters of the conceptual model will be dynamic.  
        initial_states: Optional[Dict[str, torch.Tensor]]
            Optional parameter! In case one wants to specify the initial state of the internal states of the conceptual
            model. 

        Returns
        -------
        Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]
            - y_hat: torch.Tensor
                Simulated outflow
            - parameters: Dict[str, torch.Tensor]
                Dynamic parameterization of the conceptual model
            - internal_states: Dict[str, torch.Tensor]
                Time-evolution of the internal states of the conceptual model
            - last_states: Dict[str, torch.Tensor]
                Internal states of the conceptual model in the last timestep
        """
        # initialize structures to store the information
        states, out = self._initialize_information(conceptual_inputs=x_conceptual)

        # initialize constants
        zero = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device)
        
        # Broadcast tensor to consider multiple conceptual models running in parallel
        precipitation = torch.tile(x_conceptual[:,:,0].unsqueeze(2), (1, 1, self.n_conceptual_models))
        et = torch.tile(x_conceptual[:,:,1].unsqueeze(2), (1, 1, self.n_conceptual_models))
        if x_conceptual.shape[2]==4: # the user specified tmax and tmin
            temperature = (x_conceptual[:, :, 2] + x_conceptual[:, :, 3]) / 2
        else:
            temperature = x_conceptual[:, :, 2]
        temperature = torch.tile(temperature.unsqueeze(2), (1, 1, self.n_conceptual_models))
    
        # Division between solid and liquid precipitation can be done outside of the loop
        temp_mask = temperature < parameters['TT']
        liquid_p = precipitation.clone()
        liquid_p[temp_mask] = zero
        snow = precipitation.clone()
        snow[~temp_mask] = zero
        
        if initial_states is None: # if we did not specify initial states it takes the default values
            SNOWPACK    = torch.full((x_conceptual.shape[0], self.n_conceptual_models), 
                                     self._initial_states['SNOWPACK'], dtype=torch.float32, device=x_conceptual.device)
            MELTWATER   = torch.full((x_conceptual.shape[0], self.n_conceptual_models), 
                                     self._initial_states['MELTWATER'], dtype=torch.float32, device=x_conceptual.device)
            SM          = torch.full((x_conceptual.shape[0], self.n_conceptual_models), 
                                     self._initial_states['SM'], dtype=torch.float32, device=x_conceptual.device)
            SUZ         = torch.full((x_conceptual.shape[0], self.n_conceptual_models), 
                                     self._initial_states['SUZ'], dtype=torch.float32, device=x_conceptual.device)
            SLZ         = torch.full((x_conceptual.shape[0], self.n_conceptual_models), 
                                     self._initial_states['SLZ'], dtype=torch.float32, device=x_conceptual.device)
        else: # we specify the initial states
            SNOWPACK    = initial_states['SNOWPACK']
            MELTWATER   = initial_states['MELTWATER']
            SM          = initial_states['SM']
            SUZ         = initial_states['SUZ']
            SLZ         = initial_states['SLZ']
        
        # run hydrological model for each time step
        for j in range(x_conceptual.shape[1]):
            # Snow module -----------------------------------------------------------------------------------------
            SNOWPACK = SNOWPACK + snow[:, j, :]
            melt = parameters['CFMAX'][:, j, :] * (temperature[:, j, :] - parameters['TT'][:, j, :])
            melt = torch.clamp(melt, min=0.0)
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = parameters['CFR'][:, j, :]* parameters['CFMAX'][:, j, :] * (parameters['TT'][:, j, :] - 
                                                                                     temperature[:, j, :])
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = MELTWATER - (parameters['CWH'][:, j, :] * SNOWPACK)
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil and evaporation ---------------------------------------------------------------------------------
            soil_wetness = (SM / parameters['FC'][:, j, :]) ** parameters['BETA'][:, j, :]
            soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
            recharge = (liquid_p[:, j, :] + tosoil) * soil_wetness

            SM = SM + liquid_p[:, j, :] + tosoil - recharge
            excess = SM - parameters['FC'][:, j, :]
            excess = torch.clamp(excess, min=0.0)
            SM = SM - excess
            if 'BETAET' in parameters:
                evapfactor = (SM / (parameters['LP'][:, j, :] * parameters['FC'][:, j, :]))**parameters['BETAET'][:, j, :]
            else:
                evapfactor = SM / (parameters['LP'][:, j, :] * parameters['FC'][:, j, :])
            evapfactor  = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = et[:, j, :] * evapfactor
            ETact = torch.min(SM, ETact)
            SM = torch.clamp(SM - ETact, min = 1e-5) # SM can not be zero for gradient tracking
            
            # Groundwater boxes -------------------------------------------------------------------------------------
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, parameters['PERC'][:, j, :])
            SUZ = SUZ - PERC
            Q0 = parameters['K0'][:, j, :] * torch.clamp(SUZ - parameters['UZL'][:, j, :], min=0.0)
            SUZ = SUZ - Q0
            Q1 = parameters['K1'][:, j, :] * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERC
            Q2 = parameters['K2'][:, j, :] * SLZ
            SLZ = SLZ - Q2       
            
            # Store time evolution of the internal states
            states['SNOWPACK'][:, j, :] = SNOWPACK
            states['MELTWATER'][:, j, :] = MELTWATER
            states['SM'][:, j, :] = SM
            states['SUZ'][:, j, :] = SUZ
            states['SLZ'][:, j, :] = SLZ
            
            # total outflow
            out[:, j, 0] = torch.mean(Q0 + Q1 + Q2, dim=1)  # [mm]
        
        # last states
        final_states = self._get_final_states(states=states)

        return {'y_hat': out, 'parameters': parameters, 'internal_states': states, 'final_states': final_states}

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
    def parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
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
            'BETAET': (0.3, 5.0)
            }
    