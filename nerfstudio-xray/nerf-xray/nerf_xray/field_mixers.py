"""
Field mixers for the velocity field method.

The field mixers are used to mix the forward and backward fields.
All field mixers implement the key `get_mixing_coefficient` which returns the mixing coefficient for a given position and time.
"""
from typing import Callable, Dict, Iterable, Optional, Tuple, Union, List, Type, Literal
from dataclasses import dataclass, field
from abc import abstractmethod
from math import ceil

import numpy as np
import torch
from torch import Tensor

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.configs.base_config import InstantiateConfig
from nerf_xray.deformation_fields import BsplineTemporalDeformationField3d, BsplineTemporalDeformationField3dConfig, BSplineField1d

@dataclass
class FieldMixerConfig(InstantiateConfig):
    """Configuration for deformation field instantiation"""

    _target: Type = field(default_factory=lambda: ConstantMixer)
    """target class to instantiate"""

class FieldMixer(torch.nn.Module):
    """Field mixer abstract class"""

    config: FieldMixerConfig

    def __init__(
        self,
        config: FieldMixerConfig,
        **kwargs,
    ) -> None:
        """Initialize the field mixer

        Args:
            config: configuration for the deformation field
        """
        super().__init__()
        self.config = config
    
    @abstractmethod
    def get_mixing_coefficient(self, positions: Tensor, times: Tensor, step: Optional[int]) -> Tensor:
        """Get the mixing coefficient

        Args:
            positions: positions of the points
            times: times of the points
            step: step number (optional)
        """

    @abstractmethod
    def get_mean_amplitude(self) -> float:
        """Get the mean amplitude of the field mixer"""
        pass

    @abstractmethod
    def get_std_amplitude(self) -> float:
        """Get the standard deviation of the field mixer"""
        pass
    
    @abstractmethod
    def get_mean_std_amplitude(self) -> Tuple[float, float]:
        """Get the mean and standard deviation of the field mixer"""
        pass

    @abstractmethod
    def get_stat_dict(self) -> Dict[str, float]:
        """Get the statistics of the field mixer"""
        pass
    
    @property
    def device(self):
        return next(self.parameters()).device

@dataclass 
class ConstantMixerConfig(FieldMixerConfig):
    """Configuration for constant field mixer instantiation"""

    _target: Type = field(default_factory=lambda: ConstantMixer)
    """target class to instantiate"""
    alpha: float = 0.5
    """Alpha value for the constant field mixer"""

class ConstantMixer(torch.nn.Module):
    """Constant field mixer"""

    config: ConstantMixerConfig

    def __init__(self, config: ConstantMixerConfig) -> None:
        super().__init__()
        self.config = config
        self.register_parameter('alpha', torch.nn.parameter.Parameter(torch.tensor(config.alpha)))
    
    def get_mixing_coefficient(self, positions: Tensor, times: Tensor, step: Optional[int]) -> Tensor:
        return self.alpha
    
    def get_stat_dict(self) -> Dict[str, float]:
        return {'mean_mixing_amplitude': self.alpha.item(), 'std_mixing_amplitude': 0.0}
   
@dataclass
class SpatioTemporalMixerConfig(FieldMixerConfig):
    """Configuration for deformation field instantiation"""

    _target: Type = field(default_factory=lambda: SpatioTemporalMixer)
    """target class to instantiate"""
    num_control_points: Optional[Tuple[int,int,int]] = None
    """Number of control points in each dimension"""
    weight_nn_width: int = 16
    """Width of the neural network for the weights"""
    displacement_method: Literal['neighborhood','matrix'] = 'matrix'
    """Whether to use neighborhood calculation of bsplines or assemble full matrix""" 

class SpatioTemporalMixer(FieldMixer):

    config: SpatioTemporalMixerConfig

    def __init__(
            self, 
            config: SpatioTemporalMixerConfig,
        ) -> None:
        super().__init__(config)
        self.config = config
        
        # Create the B-spline temporal deformation field
        deformation_config = BsplineTemporalDeformationField3dConfig(
            support_outside=True,
            support_range=[(-1,1), (-1,1), (-1,1)],
            num_control_points=config.num_control_points,
            weight_nn_width=config.weight_nn_width,
            weight_nn_bias=True,
            weight_nn_gain=1,
            displacement_method=config.displacement_method,
            num_components=1
        )
        self.deformation_field = deformation_config.setup()
    
    def get_mixing_coefficient(self, positions: Tensor, times: Union[Tensor, float], step: Optional[int]) -> Tensor:
        """Get the mixing coefficient using the temporal B-spline field.
        
        Args:
            positions: positions of shape [ray, nsamples, 3]
            times: times of shape [ray, nsamples, 1] or float
            step: step number (optional)
        Returns:
            Tensor: mixing coefficients of shape [ray, nsamples, 1]
        """
        # ensure positions are 2d
        shape = positions.shape[:-1]
        positions = positions.view(-1, 3)
        if isinstance(times, float):
            times = positions.new_ones(1) * times
        times = times.reshape(-1, 1)
        alpha = self.deformation_field.displacement(positions, times)
        alpha = alpha.view(*shape, 1)
        return torch.sigmoid(alpha)
    
    def get_mean_amplitude(self):
        return self.get_mean_std_amplitude()[0]
    
    def get_std_amplitude(self):
        return self.get_mean_std_amplitude()[1]
    
    def get_mean_std_amplitude(self):
        pos = 2*torch.rand(1000, 3, device=self.device) - 1
        alpha = []
        for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            _alpha = self.get_mixing_coefficient(pos, t, None).mean().item()
            alpha.append(_alpha)
        alpha = np.array(alpha)
        return alpha.mean(), alpha.std()
    
    def get_stat_dict(self) -> Dict[str, float]:
        mean_mixing_amplitude, std_mixing_amplitude = self.get_mean_std_amplitude()
        return {
            'mean_mixing_amplitude': mean_mixing_amplitude, 
            'std_mixing_amplitude': std_mixing_amplitude,
        }
    
@dataclass
class TemporalMixerConfig(FieldMixerConfig):
    """Configuration for temporal field mixer instantiation"""

    _target: Type = field(default_factory=lambda: TemporalMixer)
    """target class to instantiate"""
    num_control_points: int = 10
    """Number of control points"""

class TemporalMixer(FieldMixer):
    """Temporal mixer"""

    config: TemporalMixerConfig

    def __init__(self, config: TemporalMixerConfig) -> None:
        super().__init__(config)
        self.config = config
        self.mixing_field = BSplineField1d(
            phi_x=torch.nn.parameter.Parameter(torch.zeros(10)), 
            support_outside=True, 
            support_range=(0,1) # time range
        )
    
    def get_mixing_coefficient(self, positions: Tensor, times: Tensor, step: Optional[int]) -> Tensor:
        # times could be coming in with various shapes
        # if we only have one time, call it with a one-element tensor
        # else call on 1d view
        t = times.flatten()
        if t.unique().numel() == 1:
            alpha = self.mixing_field(t[0].view(1))[0]
        else:
            alpha = self.mixing_field(t)
            alpha = alpha.view_as(times)
        return torch.sigmoid(alpha)

    def get_mean_amplitude(self):
        t = torch.linspace(0, 1, 21, device=self.device)
        alpha = self.get_mixing_coefficient(None, t, None)
        return alpha.mean()
    
    def get_std_amplitude(self):
        t = torch.linspace(0, 1, 21, device=self.device)
        alpha = self.get_mixing_coefficient(None, t, None)
        return alpha.std()
    
    def get_mean_std_amplitude(self):
        t = torch.linspace(0, 1, 21, device=self.device)
        alpha = self.get_mixing_coefficient(None, t, None)
        return alpha.mean(), alpha.std()

    def get_stat_dict(self) -> Dict[str, float]:
        mean_mixing_amplitude, std_mixing_amplitude = self.get_mean_std_amplitude()
        return {
            'mean_mixing_amplitude': mean_mixing_amplitude, 
            'std_mixing_amplitude': std_mixing_amplitude,
        }

@dataclass
class TemporalAnnealingMixerConfig(TemporalMixerConfig):
    """Configuration for temporal annealing field mixer instantiation"""

    _target: Type = field(default_factory=lambda: TemporalAnnealingMixer)
    """target class to instantiate"""
    max_steps: int = 1000
    """Maximum steps"""
    init_slope: float = 1.0
    """Initial slope of the sigmoid"""
    final_slope: float = 1.0
    """Maximum slope of the sigmoid"""

class TemporalAnnealingMixer(TemporalMixer):
    """Temporal annealing mixer"""

    config: TemporalAnnealingMixerConfig

    def __init__(self, config: TemporalAnnealingMixerConfig) -> None:
        super().__init__(config)
        self.config = config
        self.register_buffer('slope', torch.tensor([config.init_slope]))
        self.init_step = None
        
    def update_slope(self, step: int):
        self.slope[0] = self.config.init_slope + (self.config.final_slope - self.config.init_slope) * (step - self.init_step) / self.config.max_steps

    def get_mixing_coefficient(self, positions: Tensor, times: Tensor, step: int) -> Tensor:
        """Get the mixing coefficient using the temporal B-spline field.
        
        Args:
            positions: positions of shape [ray, nsamples, 3]
            times: times of shape [ray, nsamples, 1] or float
            step: step number
        Returns:
            Tensor: mixing coefficients of shape [ray, nsamples, 1]
        """
        if self.training:
            if self.init_step is None:
                self.init_step = step
            self.update_slope(step)

        t = times.flatten()
        if t.unique().numel() == 1:
            alpha = self.mixing_field(t[0].view(1))[0]
        else:
            alpha = self.mixing_field(t)
            alpha = alpha.view_as(times)
        return torch.sigmoid(self.slope * alpha)
    
    def get_stat_dict(self) -> Dict[str, float]:
        mean_mixing_amplitude, std_mixing_amplitude = self.get_mean_std_amplitude()
        return {
            'mean_mixing_amplitude': mean_mixing_amplitude, 
            'std_mixing_amplitude': std_mixing_amplitude,
            'slope': self.slope.item(),
        }

@dataclass
class SmoothStepMixerConfig(FieldMixerConfig):
    """Configuration for smooth step field mixer instantiation"""

    _target: Type = field(default_factory=lambda: SmoothStepMixer)
    """target class to instantiate"""
    init_slope: float = 4.926
    """Initial slope of the sigmoid"""

class SmoothStepMixer(FieldMixer):
    """Smooth step mixer"""

    config: SmoothStepMixerConfig

    def __init__(self, config: SmoothStepMixerConfig) -> None:
        super().__init__(config)
        self.config = config
        self.register_parameter('slope', torch.nn.parameter.Parameter(torch.tensor(self.config.init_slope)))
        self.register_parameter('center', torch.nn.parameter.Parameter(torch.tensor(0.5)))

    def get_mixing_coefficient(self, positions: Tensor, times: Tensor, step: Optional[int]) -> Tensor:
        return torch.sigmoid(self.slope * (times - self.center))
    
    def get_stat_dict(self) -> Dict[str, float]:
        return {
            'slope': self.slope.item(),
            'center': self.center.item(),
        }


class SpatiotemporalMixingRenderer(torch.nn.Module):
    """Spatiotemporal mixing renderer"""

    def __init__(self) -> None:
        super().__init__()
        
    @classmethod
    def forward(
        cls,
        alphas: Tensor,
        ray_samples: RaySamples,
        densities: Tensor,
    ) -> Tensor:
        positions = ray_samples.frustums.get_positions() # [ray, nsamples, 3]
        # select positions within scene box (-1 to 1)
        mask = torch.all((positions >= -1) & (positions <= 1), dim=-1, keepdim=True) # [ray, nsamples, 1]
        delta_alpha = ray_samples.deltas * alphas * mask # [ray, nsamples, 1]
        # weight by density such that empty space has no contribution
        delta_alpha = delta_alpha * densities
        acc_alpha = torch.sum(delta_alpha, dim=-2) # [ray, 1]   
        return acc_alpha
