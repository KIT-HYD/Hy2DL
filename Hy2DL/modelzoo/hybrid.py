from typing import Dict, Union
import torch
import torch.nn as nn
from baseconceptualmodel import BaseConceptualModel

class Hybrid(nn.Module):
    """Wrapper to combine a deep learning model with a conceptual hydrological models. 
    
    Hybrid model in which a conceptual hydrological model is parameterized using a LSTM network

    Parameters
    ----------
    hyperparameters : Dict[str, Union[int, float, str, dict]]
        Various hyperparameters of the model
    conceptual_model: BaseConceptualModel
        Conceptual hydrological model

    """
    def __init__(self, hyperparameters: Dict[str, Union[int, float, str, dict]], conceptual_model: BaseConceptualModel):
        super().__init__()
        # General information for the model
        self.input_size_lstm = hyperparameters['input_size_lstm']
        self.hidden_size = hyperparameters['hidden_size']
        self.num_layers = hyperparameters['no_of_layers']
        
        # The warmup period can be defined by seq_length and predict_last_n, or directly
        if 'seq_length' in hyperparameters and 'predict_last_n' in hyperparameters:
            self.seq_length = hyperparameters['seq_length']
            self.predict_last_n = hyperparameters['predict_last_n']
            self.warmup_period = self.seq_length-self.predict_last_n
        elif 'warmup_period' in hyperparameters:
            self.warmup_period = hyperparameters['warmup_period']

        # Conceptual model
        self.n_conceptual_models = hyperparameters['n_conceptual_models']
        self.parameter_type = hyperparameters['parameter_type']
        self.conceptual_model = conceptual_model(n_models=self.n_conceptual_models, parameter_type=self.parameter_type)
        self.n_conceptual_model_params = len(self.conceptual_model.parameter_ranges) * self.n_conceptual_models
        
        # linear layer
        self.linear = nn.Linear(in_features=self.hidden_size, 
                                out_features=self.n_conceptual_model_params)
        # lstm
        self.lstm = nn.LSTM(input_size = self.input_size_lstm, 
                            hidden_size = self.hidden_size, 
                            batch_first = True,
                            num_layers = self.num_layers)
        
    def forward(self, x_lstm: torch.Tensor, x_conceptual: torch.Tensor):
        """Forward pass on hybrid model. 
        
        In the forward pass, each element of the batch is associated with a basin. Therefore, the conceptual model is 
        done to run multiple basins in parallel, and also multiple entities of the model at the same time. 

        Parameters
        ----------
        x_lstm: torch.Tensor
            Tensor of size [batch_size, time_steps, input_size_lstm].

        x_conceptual: torch.Tensor
            Tensor of size [batch_size, time_steps, number_inputs_conceptual]. Each element of the batch is associated 
            with a certain basin and a certain prediction period. The time_steps refer to the number of time steps 
            (e.g. days) that our conceptual model is going to be run for. The n_inputs refer to the dynamic forcings 
            used to run the conceptual model(e.g. Precipitation, Temperature...)

        Returns
        -------
        pred: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]

        """
        
        # Initialize hidden state with zeros
        batch_size = x_lstm.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=True,
                         dtype=torch.float32, device=x_lstm.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=True, 
                         dtype=torch.float32, device=x_lstm.device)
        
        # run LSTM
        lstm_out, _ = self.lstm(x_lstm, (h0, c0))
        lstm_out = self.linear(lstm_out)
        
        # map lstm output to parameters of conceptual model
        parameters_warmup, parameters_simulation = self.conceptual_model.map_parameters(lstm_out=lstm_out,
                                                                                        warmup_period=self.warmup_period) 
        # run conceptual model: warmup
        with torch.no_grad():
            pred = self.conceptual_model(x_conceptual=x_conceptual[:,:self.warmup_period,:], 
                                         parameters = parameters_warmup)
        
        # run conceptual model: simulation
        pred = self.conceptual_model(x_conceptual=x_conceptual[:,self.warmup_period:,:], 
                                     parameters=parameters_simulation, 
                                     initial_states = pred['final_states'])
        
        return pred