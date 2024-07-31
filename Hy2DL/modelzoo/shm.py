from typing import Dict, List, Optional, Tuple, Union

import torch

from Hy2DL.modelzoo.baseconceptualmodel import BaseConceptualModel


class SHM(BaseConceptualModel):
    """Modified version of the SHM [1]_ model.

    The code creates a modified version of the SHM model that can be used as a differentiable entity to create hybrid
    models. One can run multiple entities of the model at the same time.

    Parameters
    ----------
    n_models : int
        Number of model entities that will be run at the same time
    parameter_type : List[str]
        List to specify which parameters of the conceptual model will be dynamic.

    References
    ----------
    .. [1] Ehret, U., van Pruijssen, R., Bortoli, M., Loritz, R.,  Azmi, E. and Zehe, E: Adaptive clustering: reducing
        the computational costs of distributed (hydrological) modelling by exploiting time-variable similarity among
        model elements. HESS, 24, 4389-4411, doi: 10.5194/hess-24-4389-2020, 2020

    """
    default_initial_states = {
            "ss": 0.001,
            "sf": 0.001,
            "su": 0.001,
            "si": 0.001,
            "sb": 0.001,
        }
    
    parameter_ranges = {
            "dd":    (0.0, 10.0),
            "f_thr": (10.0, 60.0),
            "sumax": (20.0, 700.0),
            "beta":  (1.0, 6.0),
            "perc":  (0.0, 1.0),
            "kf":    (0.05, 0.9),
            "ki":    (0.01, 0.5),
            "kb":    (0.001, 0.2),
        }
    
    named_fluxes = [
            "qs_out",
            "qsp_out",
            "qf_in",
            "qf_out",
            "qu_in",
            "qu_out",
            "ret",
            "qi_in",
            "qi_out",
            "qb_in",
            "qb_out",
            "q_out",
        ]

    def __init__(self, n_models: int = 1, parameter_type: List[str] = None):
        super().__init__()
        self.n_conceptual_models = n_models
        self.parameter_type = self._map_parameter_type(parameter_type=parameter_type)

    def forward(
        self,
        x_conceptual: torch.Tensor,
        parameters: Dict[str, torch.Tensor],
        initial_states: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass on the SHM model.

        Parameters
        ----------
        x_conceptual: torch.Tensor
            Tensor of size [batch_size, time_steps, n_inputs]. The batch_size is associated with a certain basin and a
            certain prediction period. The time_steps refer to the number of time steps (e.g. days) that our conceptual
            model is going to be run for. The n_inputs refer to the dynamic forcings used to run the conceptual model
            (e.g. Precipitation, Temperature...)
        parameters: Dict[str, torch.Tensor]
            Dictionary with parameterization of conceptual model
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
        states, fluxes = self._initialize_information(
            conceptual_inputs=x_conceptual
        )

        # initialize constants
        zero = torch.tensor(
            0.0, dtype=torch.float32, device=x_conceptual.device
        )
        one = torch.tensor(
            1.0, dtype=torch.float32, device=x_conceptual.device
        )
        klu = torch.tensor(
            0.90, dtype=torch.float32, device=x_conceptual.device
        )  # land use correction factor [-]

        # Broadcast tensor to consider multiple conceptual models running in parallel
        precipitation = torch.tile(
            x_conceptual[:, :, 0].unsqueeze(2),
            (1, 1, self.n_conceptual_models),
        )
        et = torch.tile(
            x_conceptual[:, :, 1].unsqueeze(2),
            (1, 1, self.n_conceptual_models),
        )
        temperature = torch.tile(
            x_conceptual[:, :, 2].unsqueeze(2),
            (1, 1, self.n_conceptual_models),
        )

        # Division between solid and liquid precipitation can be done outside of the loop as temperature is given
        temp_mask = temperature < 0
        snow_melt = temperature * parameters["dd"]
        snow_melt[temp_mask] = zero
        # liquid precipitation
        liquid_p = precipitation.clone()
        liquid_p[temp_mask] = zero
        # solid precipitation (snow)
        snow = precipitation.clone()
        snow[~temp_mask] = zero
        # permanent wilting point use in ET
        pwp = (
            torch.tensor(0.8, dtype=torch.float32, device=x_conceptual.device)
            * parameters["sumax"]
        )

        if (
            initial_states is None
        ):  # if we did not specify initial states it takes the default values
            ss = torch.full(
                (x_conceptual.shape[0], self.n_conceptual_models),
                self.default_initial_states["ss"],
                dtype=torch.float32,
                device=x_conceptual.device,
            )
            sf = torch.full(
                (x_conceptual.shape[0], self.n_conceptual_models),
                self.default_initial_states["sf"],
                dtype=torch.float32,
                device=x_conceptual.device,
            )
            su = torch.full(
                (x_conceptual.shape[0], self.n_conceptual_models),
                self.default_initial_states["su"],
                dtype=torch.float32,
                device=x_conceptual.device,
            )
            si = torch.full(
                (x_conceptual.shape[0], self.n_conceptual_models),
                self.default_initial_states["si"],
                dtype=torch.float32,
                device=x_conceptual.device,
            )
            sb = torch.full(
                (x_conceptual.shape[0], self.n_conceptual_models),
                self.default_initial_states["sb"],
                dtype=torch.float32,
                device=x_conceptual.device,
            )

        else:  # we specify the initial states
            ss = initial_states["ss"]
            sf = initial_states["sf"]
            su = initial_states["su"]
            si = initial_states["si"]
            sb = initial_states["sb"]

        # run hydrological model for each time step
        for j in range(x_conceptual.shape[1]):
            # Snow module --------------------------
            qs_out = torch.minimum(ss, snow_melt[:, j, :])
            ss = ss - qs_out + snow[:, j, :]
            qsp_out = qs_out + liquid_p[:, j, :]

            # Split snowmelt+rainfall into inflow to fastflow reservoir and unsaturated reservoir ------
            qf_in = torch.maximum(zero, qsp_out - parameters["f_thr"][:, j, :])
            qu_in = torch.minimum(qsp_out, parameters["f_thr"][:, j, :])

            # Fastflow module ----------------------
            sf = sf + qf_in
            qf_out = sf * parameters["kf"][:, j, :]
            sf = sf - qf_out

            # Unsaturated zone----------------------
            psi = (su / parameters["sumax"][:, j, :]) ** parameters["beta"][:, j, :]  # [-]
            su_temp = su + qu_in * (1 - psi)
            su = torch.minimum(su_temp, parameters["sumax"][:, j, :])
            qu_out = qu_in * psi + torch.maximum(zero, su_temp - parameters["sumax"][:, j, :])  # [mm]

            # Evapotranspiration -------------------
            ktetha = su / parameters["sumax"][:, j, :]
            et_mask = su <= pwp[:, j, :]
            ktetha[~et_mask] = one
            ret = et[:, j, :] * klu * ktetha  # [mm]
            su = torch.maximum(zero, su - ret)  # [mm]

            # Interflow reservoir ------------------
            qi_in = qu_out * parameters["perc"][:, j, :]  # [mm]
            si = si + qi_in  # [mm]
            qi_out = si * parameters["ki"][:, j, :]  # [mm]
            si = si - qi_out  # [mm]

            # Baseflow reservoir -------------------
            qb_in = qu_out * (one - parameters["perc"][:, j, :])  # [mm]
            sb = sb + qb_in  # [mm]
            qb_out = sb * parameters["kb"][:, j, :]  # [mm]
            sb = sb - qb_out

            q_out = qf_out + qi_out + qb_out  # [mm]

            # Store internal states
            states["ss"][:, j, :] = ss
            states["sf"][:, j, :] = sf
            states["su"][:, j, :] = su
            states["si"][:, j, :] = si
            states["sb"][:, j, :] = sb

            # Store fluxes
            fluxes["qs_out"][:, j, :] = qs_out
            fluxes["qsp_out"][:, j, :] = qsp_out
            fluxes["qf_in"][:, j, :] = qf_in
            fluxes["qf_out"][:, j, :] = qf_out
            fluxes["qu_in"][:, j, :] = qu_in
            fluxes["qu_out"][:, j, :] = qu_out
            fluxes["ret"][:, j, :] = ret
            fluxes["qi_in"][:, j, :] = qi_in
            fluxes["qi_out"][:, j, :] = qi_out
            fluxes["qb_in"][:, j, :] = qb_in
            fluxes["qb_out"][:, j, :] = qb_out
            fluxes["q_out"][:, j, :] = q_out

        # last states
        final_states = self._get_final_states(states=states)

        return {
            "fluxes": fluxes,
            "states": states,
            "parameters": parameters,
            "final_states": final_states,
        }
