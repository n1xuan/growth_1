"""
Deformation fields.

Some deformation fields are displacement-based, while others are velocity-based.
A key class is the 3d B-spline field.
"""
from typing import Callable, Dict, Iterable, Optional, Tuple, Union, List, Type, Literal
from dataclasses import dataclass, field
from abc import abstractmethod
from math import ceil

import numpy as np
import torch
from torch import Tensor

from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.configs.base_config import InstantiateConfig

@dataclass
class DeformationFieldConfig(InstantiateConfig):
    """Configuration for deformation field instantiation"""

    _target: Type = field(default_factory=lambda: IdentityDeformationField)
    """target class to instantiate"""

class DeformationField(torch.nn.Module):
    """Deformation field abstract class"""

    config: DeformationFieldConfig

    @abstractmethod
    def __init__(
        self,
        config: DeformationFieldConfig,
        **kwargs,
    ) -> None:
        """Initialize the deformation field

        Args:
            config: configuration for the deformation field
        """

class IdentityDeformationField(torch.nn.Module):
    def __init__(self, config: DeformationFieldConfig) -> None:
        super().__init__()

    def forward(self, x: Tensor, times: Optional[Tensor]) -> Tensor:
        return x
    
    def mean_disp(self) -> float:
        return 0.0
    
    def max_disp(self) -> float:
        return 0.0

class NeuralPhiX(torch.nn.Module):
    def __init__(self, num_control_points: int = 4, depth: int = 3, width: int = 10, init_gain: float = 1e-3, bias: bool = False):
        super().__init__()
        self.W = torch.nn.Sequential(torch.nn.Linear(1, width), torch.nn.SELU())
        for _ in range(depth-1):
            self.W.append(torch.nn.Linear(width, width))
            self.W.append(torch.nn.SELU())
        lin = torch.nn.Linear(width, num_control_points, bias=bias)
        torch.nn.init.xavier_uniform_(lin.weight, gain=init_gain)
        self.W.append(lin)

    def forward(self, x):
        return self.W(x)

class NeuralFourierX(torch.nn.Module):
    def __init__(self, num_control_points: int = 4, depth: int = 3, width: int = 10, init_gain: float = 1e-3, bias: bool = False):
        super().__init__()
        self.W = torch.nn.Sequential(torch.nn.Linear(9, width), torch.nn.SELU())
        for _ in range(depth-1):
            self.W.append(torch.nn.Linear(width, width))
            self.W.append(torch.nn.SELU())
        lin = torch.nn.Linear(width, num_control_points, bias=bias)
        torch.nn.init.xavier_uniform_(lin.weight, gain=init_gain)
        self.W.append(lin)

    def forward(self, x):
        # Fourier series expansion with a constant term, 4 sines and 4 cosines
        y = x.new_zeros(x.shape[:-1], 9)
        y[..., 0] = x[..., 0]
        for i in range(1, 5):
            y[..., i] = torch.sin(np.pi * i * x[..., 0])
            y[..., i+4] = torch.cos(np.pi * i * x[..., 0])
        return self.W(y)
    
class MLPDeformationField(torch.nn.Module):
    def __init__(self, depth: int = 3, width: int = 10, support_range: Tuple[float,float] = (0,1)):
        super().__init__()

        self.support_range = support_range

        self.W = torch.nn.Sequential(torch.nn.Linear(3, width), torch.nn.SELU())
        for _ in range(depth-1):
            self.W.append(torch.nn.Linear(width, width))
            self.W.append(torch.nn.SELU())
        lin = torch.nn.Linear(width, 3, bias=False)
        torch.nn.init.xavier_uniform_(lin.weight, gain=1e-1)
        self.W.append(lin)

    def forward(self, x: torch.Tensor, times=None) -> torch.Tensor:
        return x + self.W(torch.clamp(x, self.support_range[0], self.support_range[1]))

    def mean_disp(self) -> float:
        # sample and return the mean displacement
        device = next(self.parameters()).device
        x = torch.rand(100,3, device=device)*(self.support_range[1]-self.support_range[0]) + self.support_range[0]
        return self.W(x).abs().mean().item()

    def max_disp(self) -> float:
        # sample and return the max displacement
        device = next(self.parameters()).device
        x = torch.rand(100,3, device=device)*(self.support_range[1]-self.support_range[0]) + self.support_range[0]
        return self.W(x).abs().max().item()

class BSplineField1d(torch.nn.Module):
    """1D B-spline field used for prototyping. Differentiable field now
    """   
    def __init__(
            self, 
            phi_x: Optional[Union[Tensor, torch.nn.parameter.Parameter]] = None, 
            support_outside: bool = False, 
            support_range: Optional[Tuple[float,float]] = None,
            num_control_points: Optional[int] = None,
            A_sparse: bool = False,
            displacement_method: Literal['neighborhood','matrix'] = 'matrix'
        ) -> None:
        super().__init__()
        assert phi_x is not None or num_control_points is not None
        if phi_x is not None:
            assert phi_x.ndim == 1
            assert phi_x.shape[0] > 3
            num_control_points = phi_x.shape[0]
        if support_range is None:
            # provide support over -1 to 1
            dx = 2/(num_control_points-3)
            origin = -1 - dx
        else:
            support_min, support_max = support_range
            dx = (support_max - support_min) / (num_control_points-3)
            origin = support_min - dx
        self.displacement_method = displacement_method

        self.register_parameter('phi_x', phi_x)
        self.register_buffer('num_control_points', torch.tensor(num_control_points))
        self.register_buffer('origin', torch.tensor(origin))
        self.register_buffer('dx', torch.tensor(dx))
        self.register_buffer('support_outside', torch.tensor(support_outside))
        self.register_buffer('A_sparse', torch.tensor(A_sparse))

    def displacement(self, _t: Tensor, phi_x: Optional[Union[Tensor, torch.nn.parameter.Parameter]] = None) -> Tensor:
        # # support on -1 to 1
        # _t = 0.5 * (1+_t)*(self.phi_x.shape[0]-3) * self.dx
        # xor for phi_x
        assert (phi_x is None) != (self.phi_x is None)
        if phi_x is None:
            phi_x = self.phi_x
        assert _t.ndim == 1
        t = _t - self.origin - self.dx
        x = _t.new_zeros(t.shape)
        indices = torch.floor(t/self.dx).long() 
        if not self.support_outside:
            invalid = (indices < 0) | (indices >= len(phi_x)-3)
            valid = ~invalid
            x[invalid] = torch.nan
        else:
            valid = _t.new_ones(t.shape, dtype=torch.bool)

        for i in range(4):
            inds_loc = indices[valid] + i
            if self.support_outside:
                inds_loc = torch.clamp(inds_loc, 0, len(phi_x)-1) # support outside the domain
            x[valid] += bspline(t[valid]/self.dx - indices[valid], i) * phi_x[inds_loc]

        return x
    
    def get_A_matrix(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        assert x.ndim == 1
        dx = self.dx
        nx = self.phi_x.shape[0]
        npoints = x.shape[0]
        u = (x - self.origin - dx)/dx
        ix = torch.floor(u).long()
        u = (u - ix).to(x)
        ind_x = ix[:, None] + torch.arange(4, device=ix.device)
        out_of_support = (ind_x < 0) | (ind_x >= nx)
        out_of_support = out_of_support.any(dim=1)
        ind_x = ind_x.clamp(0, nx-1)
        weights_x = torch.stack([bspline(u, i) for i in range(4)], dim=1)
        if not self.support_outside: weights_x[out_of_support] = torch.nan
        if self.A_sparse:
            idx0 = torch.repeat_interleave(torch.arange(npoints, device=x.device), 4)
            flat_index = ind_x.view(-1)
            weights_x = weights_x.view(-1)
            assert idx0.shape == flat_index.shape
            A = torch.sparse_coo_tensor(torch.vstack([idx0, flat_index]), weights_x, size=(npoints, nx), device=x.device)
        else:
            A = x.new_zeros(npoints, nx)
            A = A.scatter_add_(dim=1, index=ind_x, src=weights_x)
        return A
    
    def matrix_vector_displacement(self, x: torch.Tensor, phi_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        A = self.get_A_matrix(x)
        if phi_x is None:
            phi_x = self.phi_x
        u = A@phi_x
        return u

    def forward(self, x: torch.Tensor):
        if self.displacement_method == 'neighborhood':
            return self.displacement(x)
        elif self.displacement_method == 'matrix':
            return self.matrix_vector_displacement(x)
        else:
            raise ValueError(f'Unknown displacement method `{self.displacement_method}`')
        
class BsplineDeformationField(torch.nn.Module):
    def __init__(self, phi_x: Optional[Union[Tensor, torch.nn.parameter.Parameter]] = None, support_outside: bool = False, num_control_points: Optional[int] = None) -> None:
        super().__init__()
        self.bspline_field = BSplineField1d(phi_x, support_outside, num_control_points)
    
    def forward(self, x: Tensor) -> Tensor:
        x[...,2] += self.bspline_field.displacement(x[...,2].view(-1)).view(x.shape[:-1])
        return x
    
# @torch.jit.script
def bspline(u: torch.Tensor, i: int) -> Tensor:
    """B-spline functions.

    Using lru_cache to speed up the computation if 
    the same u and i are used multiple times.

    Args:
        u (torch.Tensor): coordinate in domain
        i (int): index of the B-spline. One of {0,1,2,3}

    Returns:
        torch.Tensor: B-spline weight
    """
    if i == 0:
        return (1 - u)**3 / 6
    elif i == 1:
        return (3*u**3 - 6*u**2 + 4) / 6
    elif i == 2:
        return (-3*u**3 + 3*u**2 + 3*u + 1) / 6
    elif i == 3:
        return u**3 / 6
    else:
        raise ValueError(f"Invalid B-spline index {i}")
    
def bspline_deriv(u: torch.Tensor, i: int) -> Tensor:
    """Derivative of B-spline functions.

    Args:
        u (torch.Tensor): coordinate in domain
        i (int): index of the B-spline. One of {0,1,2,3}

    Returns:
        torch.Tensor: B-spline derivative
    """
    if i == 0:
        return -(1 - u)**2 / 2
    elif i == 1:
        return (3*u**2 - 4*u) / 2
    elif i == 2:
        return (-3*u**2 + 2*u + 1) / 2
    elif i == 3:
        return u**2 / 2
    else:
        raise ValueError(f"Invalid B-spline index {i}")

class BSplineField3d(torch.nn.Module):
    """Cubic B-spline 3d field.
    """
    def __init__(
            self, 
            phi_x: Optional[Union[Tensor, torch.nn.parameter.Parameter]] = None,
            support_outside: bool = False,
            support_range: Optional[List[Tuple[float,float]]] = None,
            num_control_points: Optional[Tuple[int,int,int]] = None,
            A_sparse: bool = False,
            num_components: int = 3
    ) -> None:
        """Set up the B-spline field.

        Args:
            phi_x (Union[torch.Tensor, np.ndarray]): degrees of freedom 
                of the B-spline field in order [nx, ny, nz, dim]
            support_outside (bool, optional): whether to provide support
                for locations outside the control points. Defaults to False.
        """
        super().__init__()
        assert phi_x is not None or num_control_points is not None
        if phi_x is not None:
            assert phi_x.ndim == 4
            assert phi_x.shape[0] > 3 and phi_x.shape[1] > 3 and phi_x.shape[2] > 3
            num_control_points = phi_x.shape[:3]
            num_components = phi_x.shape[3]
        nx,ny,nz = num_control_points
        grid_size = np.array([nx, ny, nz])
        if support_range is None:
            # provide support for range -1 to 1 along each dimension
            spacing = 2 / (grid_size - 3)
            origin = -1 - spacing
        else:
            assert len(support_range) == 3 and all(len(r)==2 for r in support_range)
            support_min = np.array([r[0] for r in support_range])
            support_max = np.array([r[1] for r in support_range])
            spacing = (support_max - support_min) / (grid_size - 3)
            origin = support_min - spacing
        self.register_parameter('phi_x', phi_x)
        self.register_buffer('grid_size', torch.tensor(grid_size))
        self.register_buffer('origin', torch.tensor(origin))
        self.register_buffer('spacing', torch.tensor(spacing))
        self.register_buffer('support_outside', torch.tensor(support_outside))
        self.register_buffer('A_sparse', torch.tensor(A_sparse))
        self.register_buffer('num_components', torch.tensor(num_components))

    def __repr__(self) -> str:
        f = self
        if self.phi_x is None:
            phix = 'None'
        else:
            phix = self.phi_x.shape
        return f"BSplineField(phi_x={phix}, origin={f.origin}, spacing={f.spacing})\nfull support on {f.origin + f.spacing} to {f.origin + f.spacing*(f.grid_size-2)}\n"

    def displacement(
            self, x: Tensor, y: Tensor, z: Tensor, i: int, phi_x: Optional[Union[Tensor, torch.nn.parameter.Parameter]] = None, derivative: Optional[int] = None
    ) -> torch.Tensor:
        """Displacement at points x,y,z in the direction i.

        We implement support for locations beyond control points.

        Args:
            x (torch.Tensor): x-coordinates. Can be 1d or meshgrid.
            y (torch.Tensor): y-coordinates. -"-
            z (torch.Tensor): z-coordinates. -"-
            i (int): index of the displacement direction. (x=0, y=1, z=2)

        Returns:
            torch.Tensor: displacement
        """
        if phi_x is None:
            phi_x = self.phi_x
        dx, dy, dz = self.spacing
        u = (x - self.origin[0] - dx)/dx
        v = (y - self.origin[1] - dy)/dy
        w = (z - self.origin[2] - dz)/dz
        ix = torch.floor(u).long()
        iy = torch.floor(v).long()
        iz = torch.floor(w).long()
        if not self.support_outside:
            ix_nan = (ix < 0) | (ix >= self.grid_size[0]-3)
            iy_nan = (iy < 0) | (iy >= self.grid_size[1]-3)
            iz_nan = (iz < 0) | (iz >= self.grid_size[2]-3)
        u = u - ix
        v = v - iy
        w = w - iz
        T = torch.zeros_like(x, dtype=torch.float32)
        for l in range(4):
            ix_loc = torch.clamp(ix + l, 0, self.grid_size[0]-1)
            if derivative is None or derivative!=0:
                s0 = bspline(u, l)
            else:
                s0 = bspline_deriv(u, l)
            for m in range(4):
                iy_loc = torch.clamp(iy + m, 0, self.grid_size[1]-1)
                if derivative is None or derivative!=1:
                    s1 = bspline(v, m)
                else:
                    s1 = bspline_deriv(v, m)
                s0xs1 = s0 * s1
                for n in range(4):
                    iz_loc = torch.clamp(iz + n, 0, self.grid_size[2]-1)
                    if derivative is None or derivative!=2:
                        s2 = bspline(w, n)
                    else:
                        s2 = bspline_deriv(w, n)
                    T += s0xs1 * s2 * phi_x[ix_loc, iy_loc, iz_loc, i]
        # fix spacing
        if derivative is not None:
            T = T / self.spacing[derivative]
        if not self.support_outside:
            T[ix_nan | iy_nan | iz_nan] = torch.nan
        return T
    
    def vectorized_displacement(
            self, x: Tensor, y: Tensor, z: Tensor, phi_x: Optional[Union[Tensor, torch.nn.parameter.Parameter]] = None
    ) -> torch.Tensor:
        """Vectorized version of the displacement function.
        """
        if phi_x is None:
            phi_x = self.phi_x
        dx, dy, dz = self.spacing
        u = (x - self.origin[0] - dx)/dx
        v = (y - self.origin[1] - dy)/dy
        w = (z - self.origin[2] - dz)/dz
        ix = torch.floor(u).long()
        iy = torch.floor(v).long()
        iz = torch.floor(w).long()
        if not self.support_outside:
            ix_nan = (ix < 0) | (ix >= self.grid_size[0]-3)
            iy_nan = (iy < 0) | (iy >= self.grid_size[1]-3)
            iz_nan = (iz < 0) | (iz >= self.grid_size[2]-3)
        u = u - ix
        v = v - iy
        w = w - iz
        T = x.new_zeros(*x.shape, phi_x.shape[-1])
        u_shape = u.shape
        for l in range(4):
            ix_loc = torch.clamp(ix + l, 0, self.grid_size[0]-1)
            s0 = bspline(u, l).view(*u_shape,1)
            for m in range(4):
                iy_loc = torch.clamp(iy + m, 0, self.grid_size[1]-1)
                s1 = bspline(v, m).view(*u_shape,1)
                s0xs1 = s0 * s1
                for n in range(4):
                    iz_loc = torch.clamp(iz + n, 0, self.grid_size[2]-1)
                    # careful. Swapped order of axes in phi_x
                    s2 = bspline(w, n).view(*u_shape,1)
                    T[...,:] += s0xs1 * s2 * phi_x[ix_loc, iy_loc, iz_loc, :]
        if not self.support_outside:
            T[ix_nan | iy_nan | iz_nan, :] = torch.nan
        return T
    
    def get_A_matrix(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Calculate the A-matrix.

        Displacements are then given by
        ..math::
            u(x) = A(x) \\phi_x
        where A(x) is the 2d matrix of B-spline weights evaluated 
        at x and \\phi_x is the 1d vector of B-spline degrees of freedom. 
        The matrix can be used to calculate the displacement at x 
        or to infer weights from displacements.

        Args:
            x (torch.Tensor): x-position. Must be 1d. Shape [npoints]
            y (torch.Tensor): y-position. -"-
            z (torch.Tensor): z-position. -"-

        Returns:
            torch.Tensor: Matrix of shape [npoints, nx*ny*nz]
        """
        assert x.ndim == 1 and y.ndim == 1 and z.ndim == 1
        assert x.shape[0] == y.shape[0] == z.shape[0]
        dx, dy, dz = self.spacing
        nx, ny, nz = self.grid_size
        npoints = x.shape[0]
        u = (x - self.origin[0] - dx)/dx
        v = (y - self.origin[1] - dy)/dy
        w = (z - self.origin[2] - dz)/dz
        ix = torch.floor(u).long()
        iy = torch.floor(v).long()
        iz = torch.floor(w).long()
        u = (u - ix).to(x)
        v = (v - iy).to(x)
        w = (w - iz).to(x)
        ind_x = (ix[:, None] + torch.arange(4, device=ix.device)).repeat_interleave(16, dim=1)
        ind_y = (iy[:, None] + torch.arange(4, device=iy.device)).repeat(1, 4).repeat_interleave(4, dim=1)
        ind_z = (iz[:, None] + torch.arange(4, device=iz.device)).repeat(1, 16)
        out_of_support = (ind_x < 0) | (ind_x >= nx) | (ind_y < 0) | (ind_y >= ny) | (ind_z < 0) | (ind_z >= nz)
        out_of_support = out_of_support.any(dim=1)
        ind_x = ind_x.clamp(0, nx-1)
        ind_y = ind_y.clamp(0, ny-1)
        ind_z = ind_z.clamp(0, nz-1)
        flat_index = ind_z + nz*ind_y + nz*ny*ind_x
        weights_x = torch.stack([bspline(u, i) for i in range(4)], dim=-1)
        weights_y = torch.stack([bspline(v, i) for i in range(4)], dim=-1)
        weights_z = torch.stack([bspline(w, i) for i in range(4)], dim=-1)
        weights = (weights_x[...,:,:,None,None] * weights_y[...,:,None,:,None] * weights_z[...,:,None,None,:]).reshape(npoints, 64)
        del u, v, w, weights_x, weights_y, weights_z, ind_x, ind_y, ind_z, ix, iy, iz
        if not self.support_outside: weights[out_of_support] = torch.nan

        if self.A_sparse:
            idx0 = torch.arange(npoints, device=x.device).reshape(-1, 1).repeat(1, 64).reshape(1, -1)
            flat_index = flat_index.reshape(1, -1)
            weights = weights.flatten()
            assert idx0.shape == flat_index.shape
            A = torch.sparse_coo_tensor(torch.vstack([idx0, flat_index]), weights, size=(npoints, nx*ny*nz), device=x.device)
        else:
            A = x.new_zeros(npoints, nx*ny*nz)
            A = A.scatter_add_(dim=-1, index=flat_index, src=weights)
        return A

    def matrix_vector_displacement(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, phi_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert x.shape==y.shape==z.shape
        shape = x.shape
        A = self.get_A_matrix(x.view(-1), y.view(-1), z.view(-1))
        if phi_x is None:
            phi_x = self.phi_x
        u = A@phi_x.view(-1, self.num_components)
        return u.view(*shape, self.num_components)

    def mean_disp(self, phi_x: Optional[Union[Tensor, torch.nn.parameter.Parameter]] = None):
        if phi_x is None:
            phi_x = self.phi_x
        return phi_x.abs().mean().item()
    
    def max_disp(self, phi_x: Optional[Union[Tensor, torch.nn.parameter.Parameter]] = None):
        if phi_x is None:
            phi_x = self.phi_x
        return phi_x.abs().max().item()
    
@dataclass
class BsplineDeformationField3dConfig(DeformationFieldConfig):
    """Configuration for deformation field instantiation"""

    _target: Type = field(default_factory=lambda: BsplineDeformationField3d)
    """target class to instantiate"""
    phi_x: Optional[Union[Tensor, torch.nn.parameter.Parameter]] = None
    """Tensor of control point weights"""
    support_outside: bool = True
    """Support outside the control points"""
    support_range: Optional[List[Tuple[float,float]]] = None
    """Support range for the deformation field"""
    num_control_points: Optional[Tuple[int,int,int]] = None
    """Number of control points in each dimension"""
    displacement_method: Literal['neighborhood','matrix'] = 'matrix'
    """Whether to use neighborhood calculation of bsplines or assemble full matrix""" 

class BsplineDeformationField3d(torch.nn.Module):

    config: BsplineDeformationField3dConfig

    def __init__(
            self, 
            config: BsplineDeformationField3dConfig,
        ) -> None:
        super().__init__()
        self.config = config
        if config.phi_x is None:
            assert config.num_control_points is not None
            phi_x = torch.nn.parameter.Parameter(0.01*torch.randn(*config.num_control_points, 3))
        self.bspline_field = BSplineField3d(phi_x, config.support_outside, config.support_range, config.num_control_points)
        self.warning_printed = False
        if config.displacement_method=='matrix':
            self.disp_func = self.bspline_field.matrix_vector_displacement
        elif config.displacement_method=='neighborhood':
            self.disp_func = self.bspline_field.vectorized_displacement
        else:
            raise ValueError('Displacement method has to be matrix or neighborhood')
    
    def forward(self, positions: Tensor, times: Optional[Tensor] = None) -> Tensor:
        # positions, times of shape [ray, nsamples, 3]
        displacement = positions.new_zeros(positions.shape)
        uq_times = torch.unique(times)
        assert torch.all((uq_times==0)|(uq_times==1))
        mask = (times == 1.0).squeeze()
        x = positions[mask]
        x0, x1, x2 = x[:,0], x[:,1], x[:,2]
        u = self.disp_func(x0, x1, x2)
        if u.dtype!=displacement.dtype:
            displacement = displacement.to(u)
            if not self.warning_printed:
                print('displacement dtype changed to', u.dtype)
                self.warning_printed = True
        displacement[mask] = u
        return positions + displacement
    
    def mean_disp(self) -> float:
        return self.bspline_field.mean_disp()
    
    def max_disp(self) -> float:
        return self.bspline_field.max_disp()
    
@dataclass
class BsplineTemporalDeformationField3dConfig(DeformationFieldConfig):
    """Configuration for deformation field instantiation"""

    _target: Type = field(default_factory=lambda: BsplineTemporalDeformationField3d)
    """target class to instantiate"""
    phi_x: Optional[Union[Tensor, torch.nn.parameter.Parameter]] = None
    """Tensor of control point weights"""
    support_outside: bool = True
    """Support outside the control points"""
    support_range: Optional[List[Tuple[float,float]]] = None
    """Support range for the deformation field"""
    num_control_points: Optional[Tuple[int,int,int]] = None
    """Number of control points in each dimension"""
    weight_nn_width: int = 16
    """Width of the neural network for the weights"""
    weight_nn_bias: bool = False
    """Whether to use bias in the neural network for the weights"""
    weight_nn_gain: float = 1e-3
    """Initialization gain for the weights of the final layer"""
    displacement_method: Literal['neighborhood','matrix'] = 'matrix'
    """Whether to use neighborhood calculation of bsplines or assemble full matrix""" 
    num_components: int = 3
    """Number of components in the deformation field"""

class BsplineTemporalDeformationField3d(torch.nn.Module):

    config: BsplineTemporalDeformationField3dConfig

    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(
            self, 
            config: BsplineTemporalDeformationField3dConfig,
        ) -> None:
        super().__init__()
        self.config = config
        phi_x = config.phi_x
        support_outside = config.support_outside
        support_range = config.support_range
        num_control_points = config.num_control_points
        weight_nn_width = config.weight_nn_width
        
        assert phi_x is None
        assert num_control_points is not None
        self.phi_x = None
        self.weight_nn = NeuralPhiX(
            num_control_points=self.config.num_components*np.prod(num_control_points), 
            depth=3, 
            width=weight_nn_width, 
            init_gain=config.weight_nn_gain, 
            bias=config.weight_nn_bias
        )
        self.bspline_field = BSplineField3d(
            support_outside=support_outside, 
            support_range=support_range, 
            num_control_points=num_control_points, 
            num_components=config.num_components
        )
        self.register_buffer('support_range', torch.tensor(support_range))
        self.warning_printed = False
        if config.displacement_method=='matrix':
            self.disp_func = self.bspline_field.matrix_vector_displacement
        elif config.displacement_method=='neighborhood':
            self.disp_func = self.bspline_field.vectorized_displacement
        else:
            raise ValueError('Displacement method has to be matrix or neighborhood')

    def get_u(self, positions, times):
        phi = self.weight_nn(times).view(*self.bspline_field.grid_size, self.config.num_components)
        x0, x1, x2 = positions[:,0], positions[:,1], positions[:,2]
        u = self.disp_func(x0, x1, x2, phi_x=phi)
        return u

    def forward(self, positions: Tensor, times: Tensor) -> Tensor:
        # positions of shape [ray, nsamples, 3]
        # times of shape [ray, nsamples, 1]
        use_vmap = False
        if use_vmap:
            nsamples = positions.size(-2)
            assert positions.size(-1)==3
            orig_shape = positions.shape
            
            assert times.std(dim=-2).max()==0
            times = times[..., 0, :] # [rays, 1]
            displacement = torch.vmap(self.get_u, in_dims=0, out_dims=0)(positions.view(-1, nsamples, 3), times.view(-1,1))
            displacement = displacement.reshape(orig_shape)
        else:
            displacement = positions.new_zeros(positions.shape)
            uq_times = torch.unique(times)
            for t in uq_times:
                mask = (times == t).squeeze()
                if mask.numel()==1:
                    x = positions
                else:
                    x = positions[mask]
                u = self.get_u(x, t.view(-1,1))
                if u.dtype!=displacement.dtype:
                    displacement = displacement.to(u)
                    if not self.warning_printed:
                        print('displacement dtype changed to', u.dtype)
                        self.warning_printed = True
                displacement[mask] = u
        return positions + displacement

    def displacement(self, positions: Tensor, times: Tensor) -> Tensor:
        # positions of shape [ray, nsamples, 3]
        # times of shape [ray, nsamples, 1]
        use_vmap = False
        if use_vmap:
            nsamples = positions.size(-2)
            assert positions.size(-1)==3
            orig_shape = positions.shape
            
            assert times.std(dim=-2).max()==0
            times = times[..., 0, :] # [rays, 1]
            displacement = torch.vmap(self.get_u, in_dims=0, out_dims=0)(positions.view(-1, nsamples, 3), times.view(-1,1))
            displacement = displacement.reshape(orig_shape)
        else:
            displacement = positions.new_zeros(*positions.shape[:-1], self.config.num_components)
            uq_times = torch.unique(times)
            for t in uq_times:
                mask = (times == t).squeeze()
                if mask.numel()==1:
                    x = positions
                else:
                    x = positions[mask]
                u = self.get_u(x, t.view(-1,1))
                if u.dtype!=displacement.dtype:
                    displacement = displacement.to(u)
                    if not self.warning_printed:
                        print('displacement dtype changed to', u.dtype)
                        self.warning_printed = True
                displacement[mask] = u
        return displacement

    def mean_disp(self) -> float:
        # sample and return the mean displacement
        device = next(self.parameters()).device
        t = torch.linspace(0, 1, 101, device=device).view(-1,1)
        return self.weight_nn(t).abs().mean().item()

    def max_disp(self) -> float:
        # sample and return the max displacement
        device = next(self.parameters()).device
        t = torch.linspace(0, 1, 101, device=device).view(-1,1)
        return self.weight_nn(t).abs().max().item()
    
@dataclass
class BsplineTemporalVelocityField3dConfig(BsplineTemporalDeformationField3dConfig):
    _target: Type = field(default_factory=lambda: BsplineTemporalVelocityField3d)
    """target class to instantiate"""

class BsplineTemporalVelocityField3d(BsplineTemporalDeformationField3d):
    def forward(self, positions, times):
        # positions, times of shape [ray, nsamples, 3]
        displacement = positions.new_zeros(positions.shape)
        uq_times = torch.unique(times)
        for t in uq_times:
            mask = (times == t).squeeze()
            if mask.numel()==1:
                x = positions
            else:
                x = positions[mask]
            x0, x1, x2 = x[:,0], x[:,1], x[:,2]
            if self.phi_x is None:
                phi = self.weight_nn(t.view(-1,1)).view(*self.bspline_field.grid_size, 3)
            else:
                phi = self.phi_x[:t+1].sum(dim=0)
            u = self.disp_func(x0, x1, x2, phi_x=phi)
            if u.dtype!=displacement.dtype:
                displacement = displacement.to(u)
                if not self.warning_printed:
                    print('displacement dtype changed to', u.dtype)
                    self.warning_printed = True
            displacement[mask] = u
        return displacement
    
@dataclass
class BsplineTemporalIntegratedVelocityField3dConfig(BsplineTemporalDeformationField3dConfig):
    _target: Type = field(default_factory=lambda: BsplineTemporalIntegratedVelocityField3d)
    """target class to instantiate"""
    timedelta: float = 0.05
    """Time step for integration"""

class BsplineTemporalIntegratedVelocityField3d(BsplineTemporalDeformationField3d):
    def velocity(self, x0, x1, x2, time):
        phi = self.weight_nn(time.view(-1,1)).view(*self.bspline_field.grid_size, 3)
        u = self.disp_func(x0, x1, x2, phi_x=phi)
        return u

    def forward(self, positions, times, final_time):
        # positions, times of shape [ray, nsamples, 3]
        new_pos = positions.new_zeros(positions.shape)
        uq_times = torch.unique(times)
        for t in uq_times:
            mask = (times == t).squeeze()
            if mask.numel()==1:
                x = positions.clone()
            else:
                x = positions[mask].clone()
            x0, x1, x2 = x[:,0], x[:,1], x[:,2]
            assert self.phi_x is None
            if t.item()!=final_time: # add more randomness to this to make it "grid-free"
                num_steps = ceil(torch.abs(t-final_time).item()/self.config.timedelta)
                _times = torch.linspace(t, final_time, num_steps, device=x.device)
                if _times.shape[0]>2 and self.training:
                    r = 2*(torch.rand(_times.shape[0]-2, device=x.device)-0.5) # uniform random -1 to 1
                    _times[1:-1] += 0.1*self.config.timedelta*r # perturb by up to 10% dt
                for it, _t in enumerate(_times[:-1]):
                    dt = _times[it+1] - _t
                    u = self.velocity(x0, x1, x2, _t)
                    x0, x1, x2 = x0 + u[:,0]*dt, x1 + u[:,1]*dt, x2 + u[:,2]*dt
                
            if x0.dtype!=new_pos.dtype:
                new_pos = new_pos.to(u)
                if not self.warning_printed:
                    print('displacement dtype changed to', u.dtype)
                    self.warning_printed = True
            new_pos[mask] = torch.stack([x0, x1, x2], dim=-1)
        return new_pos

class BsplineTemporalDeformationField1d(torch.nn.Module):
    def __init__(
            self, 
            phi_x: Optional[Union[Tensor, torch.nn.parameter.Parameter]]=None, 
            support_outside: bool = False, 
            support_range: Optional[Tuple[float,float]]=None,
            num_control_points: Optional[int]=None
        ) -> None:
        super().__init__()
        if phi_x is not None:
            assert phi_x.ndim == 2 # [ntimes, nz]
            num_control_points = phi_x.shape[-1]
            self.phi_x = phi_x
        else:
            assert num_control_points is not None
            self.phi_x = None
            self.weight_nn = NeuralPhiX(num_control_points, 3, 16)
        self.bspline_field = BSplineField1d(support_outside=support_outside, support_range=support_range, num_control_points=num_control_points)

    def forward(self, positions: Tensor, times: Tensor) -> Tensor:
        # positions, times of shape [ray, nsamples, 3]
        x0, x1, x2 = positions[...,0].view(-1), positions[...,1].view(-1), positions[...,2].view(-1)
        uq_times = torch.unique(times)
        assert len(uq_times)==1
        if self.phi_x is None:
            phi = self.weight_nn(uq_times[0].view(-1,1)).view(-1)
        else:
            phi = self.phi_x[:uq_times[0]+1].sum(dim=0)
        out = positions.clone()
        out[...,2] += self.bspline_field.displacement(x2, phi_x=phi).view(positions.shape[:-1])
        return out
    
class AffineTemporalDeformationField(torch.nn.Module):
    def __init__(self, A: Union[Tensor, torch.nn.parameter.Parameter]) -> None:
        super().__init__()
        self.A = A
    
    def forward(self, x: Tensor, times: Optional[Tensor]) -> Tensor:
        uq_times = torch.unique(times)
        assert len(uq_times)==1
        try:
            _A = self.A[:uq_times[0]+1].sum(dim=0)
            return x + x @ _A.T
        except TypeError:
            return x
    
class ComposedDeformationField(torch.nn.Module):
    def __init__(self, deformation_fields: Iterable[Callable]) -> None:
        super().__init__()
        self.deformation_fields = deformation_fields
    
    def forward(self, x: Tensor, times: Optional[Tensor]) -> Tensor:
        for field in self.deformation_fields:
            x = field(x, times)
        return x