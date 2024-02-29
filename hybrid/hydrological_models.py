from typing import List, Dict, Tuple, Union
import torch
import torch.nn as nn

class BaseConceptualModel(nn.Module):
    """Abstract base model class, don't use this class for model training!

    The purpose is to have some common operations that all conceptual models will need. 
    """

    def __init__(self,):
        super(BaseConceptualModel, self).__init__()


    def forward(self, x_conceptual: torch.Tensor, lstm_out: torch.Tensor) -> Dict[str, Union[torch.Tensor, 
                                                                                             Dict[str, torch.Tensor]]]:
        raise NotImplementedError

    
    def _map_parameters_conceptual(self, lstm_out: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Map the output of the data-driven part of the predefined ranges of the conceptual model that is being used.

        Parameters
        ----------
        lstm_out: torch.Tensor
            Tensor of size [batch_size, time_steps, n_param] that will be mapped to the predefined ranges of the
            conceptual model parameters to act as the dynamic parameterization.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dynamic parameterization of the conceptual model.
        """
        # Broadcast tensor to consider multiple conceptual models running in parallel
        lstm_out = lstm_out.view(lstm_out.shape[0], lstm_out.shape[1], -1, self.n_models)
        
        parameters = {}
        for index, (parameter_name, parameter_range) in enumerate(self._parameter_ranges.items()):
            range_t = torch.tensor(parameter_range, dtype=torch.float32, device=lstm_out.device)
            range_t = range_t.unsqueeze(dim=1).unsqueeze(dim=2)
            
            # Dynamic parameters
            if self.parameter_type[parameter_name]=='dynamic':
                parameters[parameter_name] = range_t[:1,:,:] + torch.sigmoid(lstm_out[:, :, index, :]) * (range_t[1:,:,:] - range_t[:1,:,:])
            # Static parameters
            elif self.parameter_type[parameter_name]=='static':
                static_parameter = lstm_out[:, -1:, index, :]
                static_parameter = static_parameter.expand_as(lstm_out[:, :, index, :])
                parameters[parameter_name] = range_t[:1,:,:] + torch.sigmoid(static_parameter) * (range_t[1:,:,:] - range_t[:1,:,:])

        return parameters

    
    def _initialize_information(self, conceptual_inputs: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Initialize the structures to store the time evolution of the internal states and the outflow of the 
        conceptual model

        Parameters
        ----------
        conceptual_inputs: torch.Tensor
            Inputs of the conceptual model

        Returns
        -------
        Tuple[Dict[str, torch.Tensor], torch.Tensor]
            - states: Dict[str, torch.Tensor]
                Dictionary to store the time evolution of the internal states (buckets) of the conceptual model
            - q_out: torch.Tensor
                Tensor to store the outputs of the conceptual model
        """

        states = {}
        # initialize dictionary to store the evolution of the states
        for name, _ in self._initial_states.items():
            states[name] = torch.zeros((conceptual_inputs.shape[0], conceptual_inputs.shape[1], self.n_models), 
                                       dtype=torch.float32, device=conceptual_inputs.device)

        # initialize vectors to store the evolution of the outputs
        out = torch.zeros((conceptual_inputs.shape[0], conceptual_inputs.shape[1], self.output_size), 
                          dtype=torch.float32, device=conceptual_inputs.device)

        return states, out
    
    @property
    def _initial_states(self)-> Dict[str, float]:
        raise NotImplementedError

    @property
    def _parameter_ranges(self) -> Dict[str, List[float]]:
        raise NotImplementedError


class bucket(BaseConceptualModel):
    """Linear reservoir model.
        
        Linear reservoir model that can be used as a differentiable entity to create hybrid models. One can run it in
        parallel for multiple basins, and also multiple entities of the model at the same time.

        Parameters
        ----------
        n_models : int
            Number of model entities that will be run at the same time
        parameter_type: Dict[str, str]
            Indicate if the model parameters will be static of dynamic (in time).

    """
    def __init__(self, n_models:int, parameter_type:Dict[str, str]):
        
        super(bucket, self).__init__()
        self.n_models = n_models
        self.parameter_type = parameter_type
        self.output_size = 1

        # To have access to properties even in we use precompilation
        self.initial_states = self._initial_states
        self.parameter_ranges = self._parameter_ranges
        
    def forward(self, x_conceptual: torch.Tensor, lstm_out: torch.Tensor) -> Dict[str, Union[torch.Tensor, 
                                                                                             Dict[str, torch.Tensor]]]:
        """Perform a forward pass on the linear reservoir. In the forward pass, each element of the batch is associated
        with a basin. Therefore, the conceptual model is done to run multiple basins in parallel, and also multiple
        entities of the model at the same time. 

        Parameters
        ----------
        x_conceptual: torch.Tensor
            Tensor of size [batch_size, time_steps, n_inputs]. The batch_size is associated with a certain basin and a
            certain prediction period. The time_steps refer to the number of time steps (e.g. days) that our conceptual
            model is going to be run for. The n_inputs refer to the dynamic forcings used to run the conceptual model
            (e.g. Precipitation, Temperature...)

        lstm_out: torch.Tensor
            Tensor of size [batch_size, time_steps, n_parameters*n_models]. The tensor comes from the data-driven model  
            and will be used to obtained the parameterization of the conceptual model. 

        Returns
        -------
        Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
            - y_hat: torch.Tensor
                Simulated outflow
            - parameters: Dict[str, torch.Tensor]
                Dynamic parameterization of the conceptual model
            - internal_states: Dict[str, torch.Tensor]]
                Time-evolution of the internal states of the conceptual model

        """
        # get model parameters
        parameters = self._map_parameters_conceptual(lstm_out=lstm_out)

        # initialize structures to store the information
        states, out = self._initialize_information(conceptual_inputs=x_conceptual)

        # Storages
        si = torch.full((x_conceptual.shape[0], self.n_models), self.initial_states['si'], dtype=torch.float32, 
                        device=x_conceptual.device)

        # run hydrological model for each time step
        for j in range(x_conceptual.shape[1]):
           
            # Broadcast tensor to consider multiple conceptual models running in parallel
            p = torch.tile(x_conceptual[:,j,0].unsqueeze(1), (1, self.n_models))
            et = torch.tile(x_conceptual[:,j,1].unsqueeze(1), (1, self.n_models))
            
            # 1 bucket reservoir ------------------
            si = si + p #[mm]
            ret = et * parameters['aux_ET'][:, j, :] #[mm]
            si = torch.maximum(torch.tensor(0.0, requires_grad=True, dtype=torch.float32), si - ret) #[mm]
            qi_out = si / parameters['ki'][:, j, :] #[mm]
            si = si - qi_out #[mm]

            # states
            states['si'][:, j, :] = si
            
            # discharge
            out[:, j, 0] = torch.mean(qi_out, dim=1)  # [mm]

        return {'y_hat': out, 'parameters': parameters, 'internal_states': states}
 
    @property
    def _initial_states(self) -> Dict[str, float]:
        return {
            'si'  : 5.0,
            }
    
    @property
    def _parameter_ranges(self) -> Dict[str, List[float]]:
        return {
            'ki'  : [1.0, 500.0],
            'aux_ET'  : [0.0, 1.5]
            }
    

class SHM(BaseConceptualModel):
    """Modified version of the SHM [#]_  model that can be used as a differentiable entity to create hybrid models. One
    can run it in parallel for multiple basins, and also multiple entities of the model at the same time.

    Parameters
    ----------
    n_models : int
        Number of model entities that will be run at the same time
    parameter_type: Dict[str, str]
        Indicate if the model parameters will be static of dynamic (in time).

    References
    ----------
    .. [#] Ehret, U., van Pruijssen, R., Bortoli, M., Loritz, R.,  Azmi, E. and Zehe, E: Adaptive clustering: reducing 
        the computational costs of distributed (hydrological) modelling by exploiting time-variable similarity among 
        model elements. HESS, 24, 4389-4411, doi: 10.5194/hess-24-4389-2020, 2020
    
    """
    def __init__(self, n_models:int, parameter_type:Dict[str, str]):
        
        super(SHM, self).__init__()
        self.n_models = n_models
        self.parameter_type = parameter_type
        self.output_size = 1

        # To have access to properties even in we use precompilation
        self.initial_states = self._initial_states
        self.parameter_ranges = self._parameter_ranges
    

    def forward(self, x_conceptual: torch.Tensor, lstm_out: torch.Tensor) -> Dict[str, Union[torch.Tensor, 
                                                                                             Dict[str, torch.Tensor]]]:
        """Perform a forward pass on the SHM model. In the forward pass, each element of the batch is associated
        with a basin. Therefore, the conceptual model is done to run multiple basins in parallel, and also multiple
        entities of the model at the same time. 

        Parameters
        ----------
        x_conceptual: torch.Tensor
            Tensor of size [batch_size, time_steps, n_inputs]. The batch_size is associated with a certain basin and a
            certain prediction period. The time_steps refer to the number of time steps (e.g. days) that our conceptual
            model is going to be run for. The n_inputs refer to the dynamic forcings used to run the conceptual model
            (e.g. Precipitation, Temperature...)

        lstm_out: torch.Tensor
            Tensor of size [batch_size, time_steps, n_parameters*n_models]. The tensor comes from the data-driven model  
            and will be used to obtained the dynamic parameterization of the conceptual model

        Returns
        -------
        Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
            - y_hat: torch.Tensor
                Simulated outflow
            - parameters: Dict[str, torch.Tensor]
                Dynamic parameterization of the conceptual model
            - internal_states: Dict[str, torch.Tensor]]
                Time-evolution of the internal states of the conceptual model

        """
        # get model parameters
        parameters = self._map_parameters_conceptual(lstm_out=lstm_out)

        # initialize structures to store the information
        states, out = self._initialize_information(conceptual_inputs=x_conceptual)

        # initialize constants
        zero = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device)
        one = torch.tensor(1.0, dtype=torch.float32, device=x_conceptual.device)
        klu = torch.tensor(0.90, dtype=torch.float32, device=x_conceptual.device)  # land use correction factor [-]
        
        # Broadcast tensor to consider multiple conceptual models running in parallel
        precipitation = torch.tile(x_conceptual[:,:,0].unsqueeze(2), (1, 1, self.n_models))
        et = torch.tile(x_conceptual[:,:,1].unsqueeze(2), (1, 1, self.n_models))
        temperature = torch.tile(x_conceptual[:,:,2].unsqueeze(2), (1, 1, self.n_models))

        # Division between solid and liquid precipitation can be done outside of the loop as temperature is given
        temp_mask = temperature<0
        snow_melt = temperature * parameters['dd']
        snow_melt[temp_mask] = zero
        # liquid precipitation
        liquid_p = precipitation.clone()
        liquid_p[temp_mask] = zero
        # solid precipitation (snow)
        snow = precipitation.clone()
        snow[~temp_mask] = zero
        # permanent wilting point use in ET
        pwp = torch.tensor(0.8, dtype=torch.float32, device=x_conceptual.device)* parameters['sumax']  
        
        # Storages
        ss = torch.full((x_conceptual.shape[0], self.n_models), self.initial_states['ss'], dtype=torch.float32, 
                        device=x_conceptual.device)
        sf = torch.full((x_conceptual.shape[0], self.n_models), self.initial_states['sf'], dtype=torch.float32, 
                device=x_conceptual.device)
        su = torch.full((x_conceptual.shape[0], self.n_models), self.initial_states['su'], dtype=torch.float32, 
                device=x_conceptual.device)
        si = torch.full((x_conceptual.shape[0], self.n_models), self.initial_states['si'], dtype=torch.float32, 
                device=x_conceptual.device)
        sb = torch.full((x_conceptual.shape[0], self.n_models), self.initial_states['sb'], dtype=torch.float32, 
                device=x_conceptual.device)

        # run hydrological model for each time step
        for j in range(x_conceptual.shape[1]):
            # Snow module --------------------------
            qs_out = torch.minimum(ss, snow_melt[:, j, :])
            ss = ss - qs_out + snow[:, j, :]
            qsp_out = qs_out + liquid_p[:, j, :]

            # Split snowmelt+rainfall into inflow to fastflow reservoir and unsaturated reservoir ------
            qf_in = torch.maximum(zero, qsp_out - parameters['f_thr'][:, j, :])
            qu_in = torch.minimum(qsp_out, parameters['f_thr'][:, j, :])

            # Fastflow module ----------------------
            sf = sf + qf_in
            qf_out = sf / parameters['kf'][:, j, :]
            sf = sf - qf_out

            # Unsaturated zone----------------------
            psi = (su / parameters['sumax'][:, j, :]) ** parameters['beta'][:, j, :]  # [-]
            su_temp = su + qu_in * (1 - psi)
            su = torch.minimum(su_temp, parameters['sumax'][:, j, :])
            qu_out = qu_in * psi + torch.maximum(zero, su_temp - parameters['sumax'][:, j, :])  # [mm]
            # Evapotranspiration -------------------
            ktetha = su / parameters['sumax'][:, j, :]
            et_mask = su <= pwp[:, j, :]
            ktetha[~et_mask] = one
            ret = et[:, j, :] * klu * ktetha  # [mm]
            su = torch.maximum(zero, su - ret)  # [mm]

            # Interflow reservoir ------------------
            qi_in = qu_out * parameters['perc'][:, j, :] # [mm]
            si = si + qi_in  # [mm]
            qi_out = si / parameters['ki'][:, j, :] # [mm]
            si = si - qi_out  # [mm]

            # Baseflow reservoir -------------------
            qb_in = qu_out * (one - parameters['perc'][:, j, :])  # [mm]
            sb = sb + qb_in  # [mm]
            qb_out = sb / parameters['kb'][:, j, :]  # [mm]
            sb = sb - qb_out
            
            # Store time evolution of the internal states
            states['ss'][:, j, :] = ss
            states['sf'][:, j, :] = sf
            states['su'][:, j, :] = su
            states['si'][:, j, :] = si
            states['sb'][:, j, :] = sb

            # total outflow
            out[:, j, 0] = torch.mean(qf_out + qi_out + qb_out, dim=1)  # [mm]

        return {'y_hat': out, 'parameters': parameters, 'internal_states': states}
    
 
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
    def _parameter_ranges(self) -> Dict[str, List[float]]:
        return {
            'dd': [0.0, 10.0],
            'f_thr'  : [10.0,60.0],
            'sumax'  : [20.0,700.0],
            'beta'  : [1.0, 6.0],
            'perc'  : [0.0, 1.0],
            'kf'  : [1.0, 20.0],
            'ki'  : [1.0, 100.0],
            'kb'  : [10.0, 1000.0 ]
            }