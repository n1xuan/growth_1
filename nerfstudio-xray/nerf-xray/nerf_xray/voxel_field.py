"""
Template Nerfstudio Field

Currently this subclasses the NerfactoField. Consider subclassing the base Field.
"""

from typing import Dict, Literal, Optional, Tuple, Union, Callable, Type

import torch
from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field  # for custom Field
from nerfstudio.fields.base_field import get_normalized_directions
from nerfstudio.fields.nerfacto_field import \
    NerfactoField  # for subclassing NerfactoField
from torch import Tensor
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np

@dataclass
class VoxelGridFieldConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: VoxelGridField)
    aabb: Tensor = field(default_factory=lambda: torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]))
    max_res: int = 2048
    average_init_density: float = 1.0
    voxel_grid_file: Optional[Path] = None
    trainable_grid: bool = False

class VoxelGridField(Field):
    """Template Field

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
    """

    aabb: Tensor

    def __init__(
        self,
        config: VoxelGridFieldConfig,
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", config.aabb)
        self.register_buffer("average_init_density", torch.tensor(config.average_init_density))
        if config.voxel_grid_file is not None:
            voxel_grid = self.load_grid(config.voxel_grid_file).unsqueeze(0).unsqueeze(0)
            max_res = max(voxel_grid.shape[-3:])
        else:
            voxel_grid = torch.ones(1, 1, config.max_res, config.max_res, config.max_res)
            max_res = config.max_res
        self.register_buffer("max_res", torch.tensor(max_res))
        if config.trainable_grid:
            self.voxel_grid = torch.nn.Parameter(voxel_grid)
            self.register_buffer("a", torch.tensor(1.0))
            self.register_buffer("b", torch.tensor(0.0))
        else:
            self.register_buffer("voxel_grid", voxel_grid)
            self.register_parameter("a", torch.nn.Parameter(torch.tensor(1.0)))
            self.register_parameter("b", torch.nn.Parameter(torch.tensor(0.0)))
        self.config = config

    @staticmethod
    def load_grid(path: Union[str, Path]) -> Tensor:
        path = Path(path)
        if path.suffix == ".npz":
            with np.load(path) as data:
                vol = np.swapaxes(data["vol"], 0, 2)
        else:
            assert path.suffix == ".npy", f"Expected .npy file, got {path.suffix}"
            vol = np.load(path).swapaxes(0, 2)
        vol = vol.astype(np.float32) / vol.max()
        return torch.tensor(vol, dtype=torch.float32)

    def _density(self, pos: torch.Tensor):
        # expect pos in -1 to 1 range
        # use grid_sample
        if pos.ndim==2:
            _pos = pos.view(1,1,1,-1,3)
        elif pos.ndim==3:
            _pos = pos.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {pos.ndim}D")
        rho = torch.nn.functional.grid_sample(self.voxel_grid.to(pos), _pos, align_corners=True)
        return rho.reshape(*pos.shape[:-1], 1)

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        return {}
    
    def get_density(self, ray_samples: Union[RaySamples, Tensor], deformation_field: Optional[Union[torch.nn.Module, Callable]] = None) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        positions = ray_samples.frustums.get_positions() # positions between -1 and 1
        if deformation_field is not None:
            positions = deformation_field(positions, ray_samples.times)
        positions = SceneBox.get_normalized_positions(positions, self.aabb) # positions between 0 and 1
        # positions between -1 and 1
        positions = (positions * 2.0) - 1.0
        h_to_shape = ray_samples.frustums.shape
        selector = ((positions > -1.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True

        pos_flat = positions.view(-1, 3)
        density_before_activation = self._density(pos_flat)
        density_before_activation = density_before_activation.reshape(*h_to_shape, 1)
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = density_before_activation * self.a + self.b
        # density = self.average_init_density * trunc_exp(density.to(positions))
        density = self.average_init_density * torch.nn.functional.relu(density.to(positions))
        density = density * selector[..., None]
        return density, None

    def get_density_from_pos(
        self, positions: Tensor, deformation_field: Optional[Union[torch.nn.Module, Callable]] = None, time: Optional[Union[Tensor, float]] = 0.0
    ) -> Tensor:
        if isinstance(time, Tensor):
            pass
        else:
            time = torch.tensor([time], device=positions.device)
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=positions,
                directions=torch.ones_like(positions),
                starts=torch.zeros_like(positions[..., :1]),
                ends=torch.zeros_like(positions[..., :1]),
                pixel_area=torch.ones_like(positions[..., :1]),
            ),
            times=time,
        )
        density, _ = self.get_density(ray_samples, deformation_field=deformation_field)
        return density
    
    def forward(self, ray_samples: RaySamples, compute_normals: bool = False, deformation_field: Optional[Union[torch.nn.Module, Callable]] = None) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        if compute_normals:
            with torch.enable_grad():
                density, _ = self.get_density(ray_samples, deformation_field)
        else:
            density, _ = self.get_density(ray_samples, deformation_field)

        field_outputs = self.get_outputs(ray_samples)
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore

        if compute_normals:
            with torch.enable_grad():
                normals = self.get_normals()
            field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs

class PlaceHolderField(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return None

    def get_density_from_pos(self, *args, **kwargs):
        return None