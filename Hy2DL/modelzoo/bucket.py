from typing import Dict, List, Optional, Union

import torch

from Hy2DL.modelzoo.baseconceptualmodel import BaseConceptualModel


class Bucket(BaseConceptualModel):
    default_initial_states = {"s": 0.001}
    parameter_ranges = {"k": (0.002, 1.0), "aux_ET": (0.01, 1.5)}
    named_fluxes = ["q_out", "ret"]

    def __init__(self, n_models: int = 1, parameter_type: List[str] = None):
        super(Bucket, self).__init__()
        self.n_conceptual_models = n_models
        self.parameter_type = self._map_parameter_type(
            parameter_type=parameter_type
        )

    def forward(
        self,
        x_conceptual: torch.Tensor,
        parameters: Dict[str, torch.Tensor],
        initial_states: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        # initialize structures to store the information
        states, fluxes = self._initialize_information(
            conceptual_inputs=x_conceptual
        )

        # initialize constants
        zero = torch.tensor(
            0.0, dtype=torch.float32, device=x_conceptual.device
        )

        precipitation = torch.tile(
            x_conceptual[:, :, 0].unsqueeze(2),
            (1, 1, self.n_conceptual_models),
        )
        et = torch.tile(
            x_conceptual[:, :, 1].unsqueeze(2),
            (1, 1, self.n_conceptual_models),
        )

        # Storages
        if (
            initial_states is None
        ):  # if we did not specify initial states it takes the default values
            s = torch.full(
                (x_conceptual.shape[0], self.n_conceptual_models),
                self.default_initial_states["s"],
                dtype=torch.float32,
                device=x_conceptual.device,
            )

        else:  # we specify the initial states
            s = initial_states["s"]

        # run hydrological model for each time step
        for j in range(x_conceptual.shape[1]):
            # 1 bucket reservoir ------------------
            s = s + precipitation[:, j, :]  # [mm]
            ret = et[:, j, :] * parameters["aux_ET"][:, j, :]  # [mm]
            s = torch.maximum(zero, s - ret)  # [mm]
            q_out = s * parameters["k"][:, j, :]  # [mm]
            s = s - q_out  # [mm]

            # Store internal states
            states["s"][:, j, :] = s
            
            # Store fluxes
            fluxes["ret"][:, j, :] = ret
            fluxes["q_out"][:, j, :] = q_out

        # last states
        final_states = self._get_final_states(states=states)

        return {
            "fluxes": fluxes,
            "states": states,
            "parameters": parameters,
            "final_states": final_states,
        }
