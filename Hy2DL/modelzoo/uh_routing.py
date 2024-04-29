from typing import List, Dict
import torch
import torch.nn as nn
from baseconceptualmodel import BaseConceptualModel

class UH_routing(BaseConceptualModel):
    """Unit hydrograph routing
    """
    def __init__(self, n_models: int=1, parameter_type: Dict[str, str]={}):
        super(UH_routing, self).__init__()
        self.n_models = n_models 
        self.parameter_type = self._map_parameter_type(parameter_type=parameter_type)
        self.output_size = 1


    def forward(self, discharge: torch.Tensor, parameters: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass on the routing model

        Parameters
        ----------
        discharge: torch.Tensor
            Discharge series

        parameters: Dict[str, torch.Tensor]
            Dictionary with parameterization of routing model

        Returns
        -------
        y_routed: torch.Tensor
            Discharge series after applying the rouing module
        """

        UH = self._UH_gamma(a=parameters['routa'].permute([1,0,2]), b=parameters['routb'].permute([1,0,2]), lenF=15)
        y_routed = self._UH_conv(x = discharge.permute([0,2,1]), UH=UH.permute([1,2,0]))
        return y_routed

    
    def _UH_gamma(self, a: torch.Tensor, b: torch.Tensor, lenF: int=10):
        """Unit hydrograph based on gamma function. Implemented by Dapeng Feng [#]_.

        This code was taken from https://github.com/mhpi/dPLHBVrelease and adapted for our specific case.

        UH = 1 / (gamma(alpha)*thao^alpha) * t ^(alpha-1) * exp(-1/thao) 

        Parameters
        ----------
        a: torch.Tensor
            parameter
        b: torch.Tensor
            parameter
        lenF: int
            Number of timesteps the unitary hydrograph will have

        Returns
        -------
        w: torch.Tensor
            Unit hydrograph

        References
        ----------
        .. [#] Feng, D., Liu, J., Lawson, K., & Shen, C. (2022). Differentiable, learnable, regionalized process-based 
            models with multiphysical outputs can approach state-of-the-art hydrologic prediction accuracy. Water 
            Resources Research, 58, e2022WR032404. https://doi.org/10.1029/2022WR032404
    
        """
        m = a.shape
        w = torch.zeros([lenF, m[1],m[2]])
        aa = nn.functional.relu(a[0:lenF,:,:]).view([lenF, m[1],m[2]])+0.1 # minimum 0.1. First dimension of a is repeat
        theta = nn.functional.relu(b[0:lenF,:,:]).view([lenF, m[1],m[2]])+0.5 # minimum 0.5
        t = torch.arange(0.5,lenF*1.0, device=a.device).view([lenF,1,1]).repeat([1,m[1],m[2]])
        denom = (aa.lgamma().exp())*(theta**aa)
        mid= t**(aa-1)
        right=torch.exp(-t/theta)
        w = 1/denom*mid*right
        w = w/w.sum(0) # scale to 1 for each UH

        return w
    
    def _UH_conv(self, x: torch.Tensor, UH: torch.Tensor):
        """Unitary hydrograph routing. Implemented by Dapeng Feng [#]_.

        This code was taken from https://github.com/mhpi/dPLHBVrelease and adapted for our specific case.

        Parameters
        ----------
        x: torch.Tensor
            Discharge series
        UH: torch.Tensor
            Unit hydrograph

        Returns
        -------
        y: torch.Tensor
            Routed discharge

        References
        ----------
        .. [#] Feng, D., Liu, J., Lawson, K., & Shen, C. (2022). Differentiable, learnable, regionalized process-based 
            models with multiphysical outputs can approach state-of-the-art hydrologic prediction accuracy. Water 
            Resources Research, 58, e2022WR032404. https://doi.org/10.1029/2022WR032404
        """
                
        mm= x.shape; nb=mm[0]
        m = UH.shape[-1]
        padd = m-1

        xx = x.view([1,nb,mm[-1]])
        w  = UH.view([nb,1,m])
        groups = nb

        y = nn.functional.conv1d(xx, torch.flip(w,[2]), groups=groups, padding=padd, stride=1, bias=None)
        y=y[:,:,0:-padd]

        return y.permute([1,2,0])

    @property
    def parameter_ranges(self) -> Dict[str, List[float]]:
        return {
            'routa' : [0.0, 2.9],
            'routb' : [0.0, 6.5]
            }