from typing import Dict, Union
import torch
import torch.nn as nn

class CudaLSTM(nn.Module):
    """LSTM network. 

    Parameters
    ----------
    hyperparameters : Dict[str, Union[int, float, str, dict]]
        Various hyperparameters of the model
    """
    def __init__(self, hyperparameters: Dict[str, Union[int, float, str, dict]]):
        super().__init__()
        self.input_size_lstm = hyperparameters['input_size_lstm']
        self.hidden_size = hyperparameters['hidden_size']
        self.num_layers = hyperparameters['no_of_layers']
        
        self.lstm = nn.LSTM(input_size = self.input_size_lstm, 
                            hidden_size = self.hidden_size, 
                            batch_first = True,
                            num_layers = self.num_layers)

        
        self.dropout = torch.nn.Dropout(hyperparameters['drop_out_rate'])
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=1)
           
    def forward(self, x: torch.Tensor):
        """Forward pass of lstm networj 

        Parameters
        ----------
        x_lstm: torch.Tensor
            Tensor of size [batch_size, time_steps, input_size_lstm].

        Returns
        -------
        pred: Dict[str, torch.Tensor]
        """
        # initialize hidden state with zeros
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=True, dtype=torch.float32, 
                         device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=True, dtype=torch.float32,
                         device=x.device)
        
        out, (hn_1, cn_1) = self.lstm(x, (h0, c0))
        out = out[:,-1,:] # sequence to one
        out = self.dropout(out)
        out = self.linear(out)

        return {'y_hat': out}