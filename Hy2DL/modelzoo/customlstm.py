from typing import Dict, Union
import math
from collections import defaultdict
import torch
import torch.nn as nn
from cudalstm import CudaLSTM

class customLSTM(nn.Module):
    """LSTM cell

    Parameters
    ----------
    hyperparameters : Dict[str, Union[int, float, str, dict]]
        Various hyperparameters of the model

    """
    def __init__(self, hyperparameters: Dict[str, Union[int, float, str, dict]]):

        # Run the __init__ method of CudaLSTM class
        super().__init__()
              
        self.num_layers = hyperparameters['no_of_layers']
        self._hidden_size = hyperparameters['hidden_size']
        self._input_size = hyperparameters['input_size_lstm']
        self.cell = _LSTMCell(input_size=self._input_size,
                                hidden_size=self._hidden_size)

        self.dropout = torch.nn.Dropout(hyperparameters['drop_out_rate'])
        self.linear = nn.Linear(in_features=self._hidden_size, out_features=1)
           
    def forward(self, x: torch.Tensor):
        """Forward pass of lstm network 

        Parameters
        ----------
        x_lstm: torch.Tensor
            Tensor of size [batch_size, time_steps, input_size_lstm].

        Returns
        -------
        pred: Dict[str, torch.Tensor]

        """
        
        # initialize hidden state with zeros
        batch_size = x.shape[1]
        h0 = x.data.new(batch_size, self._hidden_size).zero_()
        c0 = x.data.new(batch_size, self._hidden_size).zero_()

        hx = (h0, c0)

        output = defaultdict(list)
        for x_t in x:
            h0, c0 = hx
            cell_output = self.cell(x_t=x_t, h_0=h0, c_0=c0)

            h_x = (cell_output['h_n'], cell_output['c_n'])

            for key, cell_out in cell_output.items():
                output[key].append(cell_out)

        # stack to [batch size, sequence length, hidden size]
        pred = {key: torch.stack(val, 1) for key, val in output.items()}
        pred['y_hat'] = self.linear(self.dropout(pred['h_n']))
        return pred
    
    def copy_weights(self, optimized_lstm: CudaLSTM):
        """Copy weights from a `CudaLSTM` or `EmbCudaLSTM` into this model class

        Parameters
        ----------
        optimized_lstm : Union[CudaLSTM, EmbCudaLSTM]
            Model instance of a `CudaLSTM` (neuralhydrology.modelzoo.cudalstm) or `EmbCudaLSTM`
            (neuralhydrology.modelzoo.embcudalstm).
            
        Raises
        ------
        RuntimeError
            If `optimized_lstm` is an `EmbCudaLSTM` but this model instance was not created with an embedding network.
        """

        # copy lstm cell weights
        self.cell.copy_weights(optimized_lstm.lstm, layer=0)

        # copy weights of linear layer
        self.linear.load_state_dict(optimized_lstm.linear.state_dict())
    
class _LSTMCell(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, initial_forget_bias: float = 0.0):
        super(_LSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.initial_forget_bias = initial_forget_bias

        self.w_hh = nn.Parameter(torch.FloatTensor(4 * hidden_size, hidden_size))
        self.w_ih = nn.Parameter(torch.FloatTensor(4 * hidden_size, input_size))

        self.b_hh = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        self.b_ih = nn.Parameter(torch.FloatTensor(4 * hidden_size))

        # self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        stdv = math.sqrt(3 / self.hidden_size)
        for weight in self.parameters():
            if len(weight.shape) > 1:
                weight.data.uniform_(-stdv, stdv)
            else:
                nn.init.zeros_(weight)

        if self.initial_forget_bias != 0:
            self.b_hh.data[self.hidden_size:2 * self.hidden_size] = self.initial_forget_bias

    def forward(self, x_t: torch.Tensor, h_0: torch.Tensor, c_0: torch.Tensor) -> Dict[str, torch.Tensor]:
        gates = h_0 @ self.w_hh.T + self.b_hh + x_t @ self.w_ih.T + self.b_ih
        i, f, g, o = gates.chunk(4, 1)

        c_1 = c_0 * torch.sigmoid(f) + torch.sigmoid(i) * torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)

        return {'h_n': h_1, 'c_n': c_1, 'i': i, 'f': f, 'g': g, 'o': o}

    def copy_weights(self, cudnnlstm: nn.Module, layer: int):

        assert self.hidden_size == cudnnlstm.hidden_size
        assert self.input_size == cudnnlstm.input_size

        self.w_hh.data = getattr(cudnnlstm, f"weight_hh_l{layer}").data
        self.w_ih.data = getattr(cudnnlstm, f"weight_ih_l{layer}").data
        self.b_hh.data = getattr(cudnnlstm, f"bias_hh_l{layer}").data
        self.b_ih.data = getattr(cudnnlstm, f"bias_ih_l{layer}").data