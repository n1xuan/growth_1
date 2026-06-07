"""
Growth-aware deformation fields for dendrite 4D reconstruction.
CORRECTED VERSION — includes mean_growth/max_growth, gradient checkpointing, _ode_step.

Extends BsplineTemporalIntegratedVelocityField3d from neural_xray
with a scalar growth source term G(x,t) that accumulates density along
the ODE trajectory:
    dx/dt = v(x,t)                          (coordinate warp)
    dg/dt = G(x(t), t)                      (density growth accumulation)

Final density:
    rho(x, t) = rho_canonical(Phi(x, t->T)) - ReLU(integral_{t}^{T} G(x(tau), tau) dtau)
"""
from typing import Optional, Tuple, Type, List, Literal, Union
from dataclasses import dataclass, field
from math import ceil

import numpy as np
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from nerf_xray.deformation_fields import (
    BsplineTemporalIntegratedVelocityField3d,
    BsplineTemporalIntegratedVelocityField3dConfig,
    BsplineTemporalDeformationField3dConfig,
    BSplineField3d,
    NeuralPhiX,
)


@dataclass
class GrowthAwareVelocityField3dConfig(BsplineTemporalIntegratedVelocityField3dConfig):
    """Configuration for growth-aware velocity field.
    
    Inherits all velocity field parameters and adds growth-specific ones.
    """
    _target: Type = field(default_factory=lambda: GrowthAwareVelocityField3d)

    # Growth field parameters
    growth_num_control_points: Optional[Tuple[int, int, int]] = None
    """Control points for growth B-spline. If None, uses same as velocity."""
    growth_nn_width: int = 16
    """Width of the growth MLP."""
    growth_nn_gain: float = 1e-4
    """Init gain for growth MLP final layer. Small = start near-zero growth."""
    growth_nn_bias: bool = False
    """Whether to use bias in growth MLP."""
    growth_support_range: Optional[List[Tuple[float, float]]] = None
    """Support range for growth B-spline. If None, uses same as velocity."""
    enable_growth: bool = True
    """Master switch. Set False to fall back to pure velocity field."""
    use_gradient_checkpointing: bool = True
    """Enable gradient checkpointing for ODE chains > 10 steps."""


class GrowthAwareVelocityField3d(BsplineTemporalIntegratedVelocityField3d):
    """Velocity field with integrated growth source term.
    
    forward() returns (warped_positions, growth_accumulation) tuple
    instead of just warped_positions when enable_growth=True.
    """

    def __init__(self, config: GrowthAwareVelocityField3dConfig) -> None:
        super().__init__(config)
        self.config: GrowthAwareVelocityField3dConfig

        if not config.enable_growth:
            self.growth_nn = None
            self.growth_bspline = None
            return

        # Determine growth grid resolution
        gcp = config.growth_num_control_points
        if gcp is None:
            gcp = config.num_control_points
        
        # Determine growth support range
        g_support = config.growth_support_range
        if g_support is None:
            g_support = config.support_range

        # Growth rate network: t -> scalar B-spline control point weights
        self.growth_nn = NeuralPhiX(
            num_control_points=int(np.prod(gcp)),  # 1 component (scalar)
            depth=3,
            width=config.growth_nn_width,
            init_gain=config.growth_nn_gain,
            bias=config.growth_nn_bias,
        )

        # Growth B-spline field: scalar output
        self.growth_bspline = BSplineField3d(
            support_outside=config.support_outside,
            support_range=g_support,
            num_control_points=gcp,
            num_components=1,  # scalar growth rate
        )

        # Use same displacement method as velocity field
        if config.displacement_method == 'matrix':
            self.growth_disp_func = self.growth_bspline.matrix_vector_displacement
        elif config.displacement_method == 'neighborhood':
            self.growth_disp_func = self.growth_bspline.vectorized_displacement
        else:
            raise ValueError(f'Displacement method: {config.displacement_method}')

    def growth_rate(self, x0: Tensor, x1: Tensor, x2: Tensor, time: Tensor) -> Tensor:
        """Compute instantaneous growth rate G(x, t).
        
        Args:
            x0, x1, x2: [N] spatial coordinates
            time: scalar tensor, the current time
            
        Returns:
            [N] scalar growth rate at each position
        """
        phi_g = self.growth_nn(time.view(-1, 1)).view(
            *self.growth_bspline.grid_size, 1
        )
        g = self.growth_disp_func(x0, x1, x2, phi_x=phi_g)
        return g.squeeze(-1)  # [N, 1] -> [N]

    def _ode_step(self, x0, x1, x2, g_acc, t, dt, enable_growth):
        """Single forward Euler step for ODE integration.
        
        Factored out to enable gradient checkpointing.
        
        Step order (Lagrangian frame):
            1. Advect position: x_{k+1} = x_k + v(x_k, t) * dt
            2. Evaluate growth at advected position: G(x_{k+1}, t)
            3. Accumulate: g_{k+1} = g_k + G(x_{k+1}, t) * dt
        """
        u = self.velocity(x0, x1, x2, t)
        x0 = x0 + u[:, 0] * dt
        x1 = x1 + u[:, 1] * dt
        x2 = x2 + u[:, 2] * dt
        if enable_growth:
            g = self.growth_rate(x0, x1, x2, t)
            g_acc = g_acc + g * dt
        return x0, x1, x2, g_acc

    def forward(self, positions, times, final_time):
        """Forward Euler ODE integration with growth accumulation.
        
        Integrates from each point's time t to final_time (T=1.0).
        
        Returns:
            If enable_growth: (warped_positions [N,3], growth_accum [N])
            If not: warped_positions [N,3]
        """
        enable_growth = (self.growth_nn is not None) and self.config.enable_growth
        use_ckpt = self.config.use_gradient_checkpointing
        
        new_pos = positions.new_zeros(positions.shape)
        if enable_growth:
            growth_accum = positions.new_zeros(positions.shape[:-1])
        
        uq_times = torch.unique(times)
        for t in uq_times:
            mask = (times == t).squeeze(-1)  # [N, 1] -> [N], never squeeze to scalar
            if mask.dim() == 0:
                mask = mask.unsqueeze(0)  # scalar -> [1]
            x = positions[mask].clone()
            x0, x1, x2 = x[..., 0], x[..., 1], x[..., 2]
            
            if enable_growth:
                g_acc = x0.new_zeros(x0.shape)
            else:
                g_acc = None

            assert self.phi_x is None

            if t.item() != final_time:
                num_steps = ceil(
                    torch.abs(t - final_time).item() / self.config.timedelta
                )
                _times = torch.linspace(
                    t, final_time, num_steps, device=x.device
                )
                # Random time perturbation during training
                if _times.shape[0] > 2 and self.training:
                    r = 2 * (torch.rand(_times.shape[0] - 2, device=x.device) - 0.5)
                    _times[1:-1] += 0.1 * self.config.timedelta * r

                for it, _t in enumerate(_times[:-1]):
                    dt = _times[it + 1] - _t

                    if self.training and use_ckpt and num_steps > 10:
                        # Gradient checkpointing for long ODE chains
                        x0, x1, x2, g_acc = torch_checkpoint(
                            self._ode_step,
                            x0, x1, x2, g_acc, _t, dt, enable_growth,
                            use_reentrant=False
                        )
                    else:
                        x0, x1, x2, g_acc = self._ode_step(
                            x0, x1, x2, g_acc, _t, dt, enable_growth
                        )

            if x0.dtype != new_pos.dtype:
                new_pos = new_pos.to(x0)
                if not self.warning_printed:
                    print('displacement dtype changed to', x0.dtype)
                    self.warning_printed = True
            
            new_pos[mask] = torch.stack([x0, x1, x2], dim=-1)
            if enable_growth:
                if g_acc.dtype != growth_accum.dtype:
                    growth_accum = growth_accum.to(g_acc)
                growth_accum[mask] = g_acc

        if enable_growth:
            return new_pos, growth_accum
        else:
            return new_pos

    def mean_disp(self) -> float:
        """Mean displacement (for logging)."""
        device = next(self.parameters()).device
        t = torch.linspace(0, 1, 101, device=device).view(-1, 1)
        return self.weight_nn(t).abs().mean().item()

    def max_disp(self) -> float:
        """Max displacement (for logging)."""
        device = next(self.parameters()).device
        t = torch.linspace(0, 1, 101, device=device).view(-1, 1)
        return self.weight_nn(t).abs().max().item()

    def mean_growth(self) -> float:
        """Mean absolute growth rate across time (for logging)."""
        if self.growth_nn is None:
            return 0.0
        device = next(self.parameters()).device
        t = torch.linspace(0, 1, 11, device=device).view(-1, 1)
        return self.growth_nn(t).abs().mean().item()

    def max_growth(self) -> float:
        """Max absolute growth rate across time (for logging)."""
        if self.growth_nn is None:
            return 0.0
        device = next(self.parameters()).device
        t = torch.linspace(0, 1, 11, device=device).view(-1, 1)
        return self.growth_nn(t).abs().max().item()
