# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script for exporting NeRF into other formats.
"""


from __future__ import annotations

import json
import os
import sys
import typing
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast, Callable

import numpy as np
import open3d as o3d
import torch
import tyro
from rich.progress import track
from torch.utils.data import DataLoader, TensorDataset
from typing_extensions import Annotated, Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import \
    FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import \
    ParallelDataManager
from nerfstudio.data.datamanagers.random_cameras_datamanager import \
    RandomCamerasDataManager
from nerfstudio.data.scene_box import OrientedBox
# from nerfstudio.exporter import texture_utils, tsdf_utils
# from nerfstudio.exporter.exporter_utils import (collect_camera_poses,
#                                                 generate_point_cloud,
#                                                 get_mesh_from_filename)
from nerfstudio.exporter.marching_cubes import \
    generate_mesh_with_multires_marching_cubes
from nerfstudio.fields.sdf_field import SDFField  # noqa
from nerfstudio.models.base_model import Model
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class Exporter:
    """Export the mesh from a YML config to a folder."""

    load_config: Path
    """Path to the config YAML file."""
    output_dir: Path
    """Path to the output directory."""


def validate_pipeline(normal_method: str, normal_output_name: str, pipeline: Pipeline) -> None:
    """Check that the pipeline is valid for this exporter.

    Args:
        normal_method: Method to estimate normals with. Either "open3d" or "model_output".
        normal_output_name: Name of the normal output.
        pipeline: Pipeline to evaluate with.
    """
    if normal_method == "model_output":
        CONSOLE.print("Checking that the pipeline has a normal output.")
        origins = torch.zeros((1, 3), device=pipeline.device)
        directions = torch.ones_like(origins)
        pixel_area = torch.ones_like(origins[..., :1])
        camera_indices = torch.zeros_like(origins[..., :1])
        ray_bundle = RayBundle(
            origins=origins, directions=directions, pixel_area=pixel_area, camera_indices=camera_indices
        )
        outputs = pipeline.model(ray_bundle)
        if normal_output_name not in outputs:
            CONSOLE.print(f"[bold yellow]Warning: Normal output '{normal_output_name}' not found in pipeline outputs.")
            CONSOLE.print(f"Available outputs: {list(outputs.keys())}")
            CONSOLE.print(
                "[bold yellow]Warning: Please train a model with normals "
                "(e.g., nerfacto with predicted normals turned on)."
            )
            CONSOLE.print("[bold yellow]Warning: Or change --normal-method")
            CONSOLE.print("[bold yellow]Exiting early.")
            sys.exit(1)

@dataclass
class ExportVolumeGrid(Exporter):
    """Export as binary raw with MHD descriptor and numpy file."""
    resolution: int = 512
    """Resolution of the volume. Same in all dimensions."""
    export_dtype: Literal["uint8", "uint16", "float32"] = "uint8"
    """Data type to export the volume as."""
    fmt: Literal["raw", "npy", "npz"] = "npz"
    """Format to export the volume as."""
    target: Literal["field", "datamanager", "both"] = "field"
    """Target to plot. Either 'field', 'datamanager', or 'both'."""
    time: Optional[float] = 0.0
    """Time to evaluate the field at. Useful if deformation is time-dependent."""
    which: Optional[Literal['forward','backward','mixed']] = None
    """Evaluate forward or backward model or divergence-based mixing"""
    max_density: Optional[float] = None
    """Maximum density. If None, use the maximum density in the volume."""

    def main(self) -> None:
        dtypes = {"uint8": {"dtype": np.uint8, "met": "MET_UCHAR"}, "uint16": {"dtype": np.uint16, "met": "MET_USHORT"}, "float32": {"dtype": np.float32, "met": "MET_FLOAT"}}

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        model: Model = pipeline.model
        model.eval()

        densities = []
        distances = np.linspace(-1, 1, self.resolution)
        for i_slice in track(range(self.resolution), description="Assembling volume slices"):
            xy = pipeline.eval_along_plane(
                plane='xy', distance=distances[i_slice], 
                engine='numpy', 
                resolution=self.resolution, 
                target=self.target,
                time=self.time,
                which=self.which
            )
            densities.append(xy.squeeze())
        densities = np.stack(densities, axis=2)
        max_density = self.max_density if self.max_density is not None else densities.max()
        # densities *= (np.iinfo(dtypes[self.export_dtype]["dtype"]).max / max_density)
        # densities = densities.clip(0, np.iinfo(dtypes[self.export_dtype]["dtype"]).max)
        # densities = densities.astype(dtypes[self.export_dtype]["dtype"])
        target_dtype = dtypes[self.export_dtype]["dtype"]
        if np.issubdtype(target_dtype, np.integer):
            # 如果是整数 (uint8, uint16)，则拉伸到最大值并截断
            densities *= (np.iinfo(target_dtype).max / max_density)
            densities = densities.clip(0, np.iinfo(target_dtype).max)
        else:
            # 如果是浮点数 (float32)，只需保证密度不为负数即可
            densities = densities.clip(0.0, None)
        densities = densities.astype(target_dtype)
        print(f"Exporting {self.which} volume with shape {densities.shape} and dtype {densities.dtype} at time {self.time}")
        assert densities.shape == (self.resolution, self.resolution, self.resolution)
        
        if self.fmt=="npz":
            np.savez_compressed(self.output_dir / "volume.npz", vol=densities)
            return
        elif self.fmt=="npy":
            np.save(self.output_dir / "volume.npy", densities)
            return
        else:
            raise ValueError(f"Unknown format {self.fmt}")


@dataclass
class ExportImageStack(Exporter):
    """Export image stack."""
    resolution: int = 512
    """Lateral resolution of the images."""
    num_slices: int = 64
    """Number of slices."""
    plot_engine: Literal["matplotlib", "opencv"] = "opencv"
    """Plotting engine to use."""
    plane: Literal["xy", "xz", "yz"] = "xy"
    """Plane along which to slice."""
    target: Literal["field", "datamanager", "both"] = "field"
    """Target to plot. Either 'field', 'datamanager', or 'both'."""
    max_density: float = 1.0
    """Maximum density for the colormap."""
    times: Optional[List[float]] = field(default_factory=lambda: [0.0])
    """Times to evaluate the field at. Useful if deformation is time-dependent."""

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        model: Model = pipeline.model
        model.eval()

        distances = np.linspace(-1, 1, self.num_slices)
        for t in self.times:
            for i_slice in track(range(self.num_slices), description=f"Exporting image stack at time {t}"):
                fn = self.output_dir / f"image_t-{t:.2f}_{self.plane}-{i_slice:04d}.png"
                pipeline.eval_along_plane(
                    target=self.target,
                    plane=self.plane, 
                    distance=distances[i_slice], 
                    fn=fn, 
                    engine=self.plot_engine, 
                    resolution=self.resolution,
                    rhomax=self.max_density,
                    time=t
                )

@dataclass
class ExportDeformationField(Exporter):
    """Export deformation field."""
    resolution: int = 256
    """Resolution of spatial grid."""

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        model: Model = pipeline.model
        model.eval()

        assert hasattr(model, "deformation_field")
        assert isinstance(model.deformation_field, Callable)

        x = torch.linspace(-1, 1, self.resolution, device=model.device)
        X,Y,Z = torch.meshgrid(x,x,x, indexing='ij')
        pos = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
        pos_dataset = TensorDataset(pos)
        dataloader = DataLoader(pos_dataset, batch_size=1024*1024, shuffle=False)

        t = torch.linspace(0,1,11, device=model.device).view(-1,1)
        for ti in t:
            fn = self.output_dir / f"deformation_t_{ti.item():.2f}.npy"

            displacements = []
            for batch in track(dataloader, description=f"Computing deformations at t={ti.item()}"):
                pos1 = batch[0].clone()
                with torch.no_grad():
                    pos1 = model.deformation_field(pos1, ti)
                u = pos1 - batch[0]
                u = u.cpu().numpy()
                displacements.append(u)
            displacements = np.concatenate(displacements, axis=0).squeeze()
            np.save(fn, displacements)

@dataclass
class ExportVelocityField(Exporter):
    """Export velocity field."""
    resolution: int = 256
    """Resolution of spatial grid."""

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        model: Model = pipeline.model
        model.eval()

        assert hasattr(model, "deformation_field")
        assert isinstance(model.deformation_field, Callable)

        x = torch.linspace(-1, 1, self.resolution, device=model.device)
        X,Y,Z = torch.meshgrid(x,x,x, indexing='ij')
        pos = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1) # (N, 3)
        pos_dataset = TensorDataset(pos)
        dataloader = DataLoader(pos_dataset, batch_size=1024*1024, shuffle=False)

        t = torch.linspace(0,1,11, device=model.device).view(-1,1)
        for ti in t:
            fn = self.output_dir / f"deformation_t_{ti.item():.2f}.npz"

            velocities = []
            for batch in track(dataloader, description=f"Computing velocities at t={ti.item():.4g}"):
                pos1 = batch[0].clone()
                with torch.no_grad():
                    u = model.deformation_field.velocity(pos1[:, 0], pos1[:, 1], pos1[:, 2], ti)
                u = u.cpu().numpy()
                velocities.append(u)
            velocities = np.concatenate(velocities, axis=0).squeeze()
            velocities = velocities.reshape((self.resolution, self.resolution, self.resolution, 3))
            np.savez_compressed(fn, velocities)


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ExportVolumeGrid, tyro.conf.subcommand(name="volume-grid")],
        Annotated[ExportImageStack, tyro.conf.subcommand(name="image-stack")],
        Annotated[ExportDeformationField, tyro.conf.subcommand(name="deformation-field")],
        Annotated[ExportVelocityField, tyro.conf.subcommand(name="velocity-field")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa
