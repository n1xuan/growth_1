"""
Growth-aware deformation fields for dendrite 4D reconstruction.
v4: Shared temporal encoder (GrowFlow-inspired architecture).

Key change from v3: velocity and growth share a temporal backbone,
with separate output heads. This mirrors GrowFlow's design where
dmeans and dscales share the same spatiotemporal encoding.

Architecture:
    shared_backbone: t → temporal features [width]
    vel_head: temporal features → velocity B-spline weights [N_vel]
    growth_head: temporal features → growth B-spline weights [N_growth]

ODE system:
    dx/dt = v(x, t)     via vel_head → velocity B-spline
    dg/dt = G(x, t)     via growth_head → growth B-spline

Final density:
    rho(x, t) = rho_canonical(Phi(x, t->T)) × exp(-softplus(growth_accum))
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
    """Configuration for growth-aware velocity field."""
    _target: Type = field(default_factory=lambda: GrowthAwareVelocityField3d)

    # Growth field parameters
    growth_num_control_points: Optional[Tuple[int, int, int]] = None
    """Control points for growth B-spline. If None, uses same as velocity."""
    growth_nn_width: int = 16
    """Width of growth MLP (unused when shared encoder is active)."""
    growth_nn_gain: float = 1e-4
    """Init gain for growth head (overridden by zero-init)."""
    growth_nn_bias: bool = False
    """Whether to use bias in growth head."""
    growth_support_range: Optional[List[Tuple[float, float]]] = None
    """Support range for growth B-spline. If None, uses same as velocity."""
    enable_growth: bool = True
    """Master switch. Set False to fall back to pure velocity field."""
    use_gradient_checkpointing: bool = True
    """Enable gradient checkpointing for ODE chains."""
    use_shared_encoder: bool = True
    """Use shared temporal encoder for velocity and growth (recommended)."""


class GrowthAwareVelocityField3d(BsplineTemporalIntegratedVelocityField3d):
    """Velocity field with integrated growth source term.

    v4 architecture: shared temporal encoder with separate velocity/growth heads.
    Mirrors GrowFlow's design where all ODE derivatives share spatiotemporal encoding.
    """

    def __init__(self, config: GrowthAwareVelocityField3dConfig) -> None:
        super().__init__(config)
        self.config: GrowthAwareVelocityField3dConfig

        if not config.enable_growth:
            self.growth_nn = None
            self.growth_bspline = None
            self._use_shared_encoder = False
            return

        # Growth B-spline grid resolution
        gcp = config.growth_num_control_points
        if gcp is None:
            gcp = config.num_control_points

        # Growth B-spline support range
        g_support = config.growth_support_range
        if g_support is None:
            g_support = config.support_range

        # Growth B-spline field: scalar output
        self.growth_bspline = BSplineField3d(
            support_outside=config.support_outside,
            support_range=g_support,
            num_control_points=gcp,
            num_components=1,
        )

        # Growth displacement function
        if config.displacement_method == 'matrix':
            self.growth_disp_func = self.growth_bspline.matrix_vector_displacement
        elif config.displacement_method == 'neighborhood':
            self.growth_disp_func = self.growth_bspline.vectorized_displacement
        else:
            raise ValueError(f'Displacement method: {config.displacement_method}')

        # ============================================================
        # SHARED ENCODER (GrowFlow-inspired)
        # ============================================================
        if config.use_shared_encoder:
            vel_output_dim = config.num_components * int(np.prod(config.num_control_points))
            growth_output_dim = int(np.prod(gcp))
            width = config.weight_nn_width

            # Shared temporal backbone
            # Mirrors GrowFlow's shared encoder that feeds into
            # separate mlp_means / mlp_scales heads
            self.shared_backbone = torch.nn.Sequential(
                torch.nn.Linear(1, width), torch.nn.SELU(),
                torch.nn.Linear(width, width), torch.nn.SELU(),
            )

            # Velocity head
            self.vel_head = torch.nn.Linear(width, vel_output_dim, 
                                            bias=config.weight_nn_bias)
            torch.nn.init.xavier_uniform_(self.vel_head.weight, 
                                          gain=config.weight_nn_gain)
            if self.vel_head.bias is not None:
                self.vel_head.bias.data.zero_()

            # Growth head — zero-init (GrowFlow's init_as_identity)
            self.growth_head = torch.nn.Linear(width, growth_output_dim,
                                               bias=config.growth_nn_bias)
            self.growth_head.weight.data.zero_()
            if self.growth_head.bias is not None:
                self.growth_head.bias.data.zero_()

            # Remove parent's weight_nn to avoid confusion and save memory.
            # We override velocity() so weight_nn is never called.
            del self.weight_nn

            # Mark that we're using shared encoder (for param group separation)
            self.growth_nn = None  # not used, but checked in model code
            self._use_shared_encoder = True
        else:
            # Fallback: independent networks (v3 behavior)
            self.growth_nn = NeuralPhiX(
                num_control_points=int(np.prod(gcp)),
                depth=3,
                width=config.growth_nn_width,
                init_gain=config.growth_nn_gain,
                bias=config.growth_nn_bias,
            )
            with torch.no_grad():
                self.growth_nn.W[-1].weight.data.zero_()
                if self.growth_nn.W[-1].bias is not None:
                    self.growth_nn.W[-1].bias.data.zero_()
            self._use_shared_encoder = False

    # ================================================================
    # Velocity and growth rate — shared encoder versions
    # ================================================================
    def velocity(self, x0, x1, x2, time):
        """Instantaneous velocity field v(x, t).
        Uses shared encoder when available, falls back to weight_nn."""
        if self._use_shared_encoder:
            features = self.shared_backbone(time.view(-1, 1))
            phi = self.vel_head(features).view(
                *self.bspline_field.grid_size, 3
            )
        else:
            phi = self.weight_nn(time.view(-1, 1)).view(
                *self.bspline_field.grid_size, 3
            )
        return self.disp_func(x0, x1, x2, phi_x=phi)

    def growth_rate(self, x0, x1, x2, time):
        """Instantaneous growth rate G(x, t).
        Uses shared encoder when available."""
        if self._use_shared_encoder:
            features = self.shared_backbone(time.view(-1, 1))
            phi_g = self.growth_head(features).view(
                *self.growth_bspline.grid_size, 1
            )
        else:
            phi_g = self.growth_nn(time.view(-1, 1)).view(
                *self.growth_bspline.grid_size, 1
            )
        g = self.growth_disp_func(x0, x1, x2, phi_x=phi_g)
        return g.squeeze(-1)

    # ================================================================
    # RK4 ODE integration
    # ================================================================
    def _eval_derivs(self, x0, x1, x2, t, enable_growth):
        """Evaluate velocity and growth rate at (x, t)."""
        u = self.velocity(x0, x1, x2, t)
        dx0, dx1, dx2 = u[:, 0], u[:, 1], u[:, 2]
        if enable_growth:
            dg = self.growth_rate(x0, x1, x2, t)
        else:
            dg = torch.zeros_like(x0)
        return dx0, dx1, dx2, dg
    '''
    def _ode_step(self, x0, x1, x2, g_acc, t, dt, enable_growth):
        """Single RK4 step for ODE integration.
        4th-order Runge-Kutta: local error O(dt^5)."""
        # k1
        dx0_1, dx1_1, dx2_1, dg_1 = self._eval_derivs(x0, x1, x2, t, enable_growth)

        # k2
        t_mid = t + 0.5 * dt
        dx0_2, dx1_2, dx2_2, dg_2 = self._eval_derivs(
            x0 + 0.5*dt*dx0_1, x1 + 0.5*dt*dx1_1, x2 + 0.5*dt*dx2_1,
            t_mid, enable_growth
        )

        # k3
        dx0_3, dx1_3, dx2_3, dg_3 = self._eval_derivs(
            x0 + 0.5*dt*dx0_2, x1 + 0.5*dt*dx1_2, x2 + 0.5*dt*dx2_2,
            t_mid, enable_growth
        )

        # k4
        t_end = t + dt
        dx0_4, dx1_4, dx2_4, dg_4 = self._eval_derivs(
            x0 + dt*dx0_3, x1 + dt*dx1_3, x2 + dt*dx2_3,
            t_end, enable_growth
        )

        # Weighted combination
        x0 = x0 + (dt / 6.0) * (dx0_1 + 2*dx0_2 + 2*dx0_3 + dx0_4)
        x1 = x1 + (dt / 6.0) * (dx1_1 + 2*dx1_2 + 2*dx1_3 + dx1_4)
        x2 = x2 + (dt / 6.0) * (dx2_1 + 2*dx2_2 + 2*dx2_3 + dx2_4)
        if enable_growth:
            g_acc = g_acc + (dt / 6.0) * (dg_1 + 2*dg_2 + 2*dg_3 + dg_4)

        return x0, x1, x2, g_acc
    '''
    def _ode_step(self, x0, x1, x2, g_acc, t, dt, enable_growth):
        """Single Forward Euler step for ODE integration.
        Matches baseline neural_xray. Local error O(dt^2)."""
        dx0, dx1, dx2, dg = self._eval_derivs(x0, x1, x2, t, enable_growth)
        x0 = x0 + dt * dx0
        x1 = x1 + dt * dx1
        x2 = x2 + dt * dx2
        if enable_growth:
            g_acc = g_acc + dt * dg
        return x0, x1, x2, g_acc
        
    # ================================================================
    # Forward pass (ODE integration)
    # ================================================================
    def forward(self, positions, times, final_time):
        """ODE integration with growth accumulation."""
        enable_growth = (
            self.config.enable_growth 
            and (self._use_shared_encoder or self.growth_nn is not None)
        )
        use_ckpt = self.config.use_gradient_checkpointing

        new_pos = positions.new_zeros(positions.shape)
        if enable_growth:
            growth_accum = positions.new_zeros(positions.shape[:-1])

        uq_times = torch.unique(times)
        for t in uq_times:
            mask = (times == t).squeeze(-1)
            if mask.dim() == 0:
                mask = mask.unsqueeze(0)
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
                    t, final_time, num_steps + 1, device=x.device
                )
                if _times.shape[0] > 2 and self.training:
                    r = 2 * (torch.rand(_times.shape[0] - 2, device=x.device) - 0.5)
                    _times[1:-1] += 0.1 * self.config.timedelta * r

                for it, _t in enumerate(_times[:-1]):
                    dt = _times[it + 1] - _t

                    if self.training and use_ckpt and num_steps > 2:
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

    # ================================================================
    # Logging helpers
    # ================================================================
    def mean_disp(self) -> float:
        device = next(self.parameters()).device
        t = torch.linspace(0, 1, 101, device=device).view(-1, 1)
        if self._use_shared_encoder:
            features = self.shared_backbone(t)
            phi = self.vel_head(features)
        else:
            phi = self.weight_nn(t)
        return phi.abs().mean().item()

    def max_disp(self) -> float:
        device = next(self.parameters()).device
        t = torch.linspace(0, 1, 101, device=device).view(-1, 1)
        if self._use_shared_encoder:
            features = self.shared_backbone(t)
            phi = self.vel_head(features)
        else:
            phi = self.weight_nn(t)
        return phi.abs().max().item()

    def mean_growth(self) -> float:
        if not self.config.enable_growth:
            return 0.0
        device = next(self.parameters()).device
        t = torch.linspace(0, 1, 11, device=device).view(-1, 1)
        if self._use_shared_encoder:
            features = self.shared_backbone(t)
            phi = self.growth_head(features)
        else:
            phi = self.growth_nn(t)
        return phi.abs().mean().item()

    def max_growth(self) -> float:
        if not self.config.enable_growth:
            return 0.0
        device = next(self.parameters()).device
        t = torch.linspace(0, 1, 11, device=device).view(-1, 1)
        if self._use_shared_encoder:
            features = self.shared_backbone(t)
            phi = self.growth_head(features)
        else:
            phi = self.growth_nn(t)
        return phi.abs().max().item()