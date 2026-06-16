"""
Script for exporting NeRF into other formats.
CORRECTED VERSION for growth-aware model.

Changes from baseline:
1. ExportVolumeGrid: 'which' parameter becomes no-op for growth model (backward compat)
2. ExportVelocityField: also exports growth rate field if available
3. NEW: ExportGrowthField — dedicated growth rate export
4. ExportDeformationField: handles tuple returns from growth-aware deformation
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
    """Check that the pipeline is valid for this exporter."""
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
    which: Optional[Literal['forward', 'backward', 'mixed']] = None
    """Evaluate forward or backward model or divergence-based mixing.
    NOTE: For growth-aware model, this parameter is ignored (single canonical)."""
    max_density: Optional[float] = None
    """Maximum density. If None, use the maximum density in the volume."""

    def main(self) -> None:
        dtypes = {
            "uint8": {"dtype": np.uint8, "met": "MET_UCHAR"},
            "uint16": {"dtype": np.uint16, "met": "MET_USHORT"},
            "float32": {"dtype": np.float32, "met": "MET_FLOAT"},
        }

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
        target_dtype = dtypes[self.export_dtype]["dtype"]
        if np.issubdtype(target_dtype, np.integer):
            densities *= (np.iinfo(target_dtype).max / max_density)
            densities = densities.clip(0, np.iinfo(target_dtype).max)
        else:
            densities = densities.clip(0.0, None)
        densities = densities.astype(target_dtype)
        print(f"Exporting volume with shape {densities.shape} and dtype {densities.dtype} at time {self.time}")
        assert densities.shape == (self.resolution, self.resolution, self.resolution)

        if self.fmt == "npz":
            np.savez_compressed(self.output_dir / "volume.npz", vol=densities)
            return
        elif self.fmt == "npy":
            np.save(self.output_dir / "volume.npy", densities)
            return
        else:
            raise ValueError(f"Unknown format {self.fmt}")


@dataclass
class ExportImageStack(Exporter):
    """Export image stack."""
    resolution: int = 512
    num_slices: int = 64
    plot_engine: Literal["matplotlib", "opencv"] = "opencv"
    plane: Literal["xy", "xz", "yz"] = "xy"
    target: Literal["field", "datamanager", "both"] = "field"
    max_density: float = 1.0
    times: Optional[List[float]] = field(default_factory=lambda: [0.0])

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
    """Export deformation field.
    
    FIX: handles tuple returns from growth-aware deformation field.
    """
    resolution: int = 256

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        model: Model = pipeline.model
        model.eval()

        assert hasattr(model, "deformation_field")

        x = torch.linspace(-1, 1, self.resolution, device=model.device)
        X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
        pos = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
        pos_dataset = TensorDataset(pos)
        dataloader = DataLoader(pos_dataset, batch_size=1024 * 1024, shuffle=False)

        t = torch.linspace(0, 1, 11, device=model.device).view(-1, 1)
        for ti in t:
            fn = self.output_dir / f"deformation_t_{ti.item():.2f}.npy"

            displacements = []
            for batch in track(dataloader, description=f"Computing deformations at t={ti.item()}"):
                pos1 = batch[0].clone()
                with torch.no_grad():
                    # FIX: handle tuple return from growth-aware field
                    result = model.deformation_field(pos1, ti, 1.0)
                    if isinstance(result, tuple):
                        pos1_warped = result[0]
                    else:
                        pos1_warped = result
                u = pos1_warped - batch[0]
                u = u.cpu().numpy()
                displacements.append(u)
            displacements = np.concatenate(displacements, axis=0).squeeze()
            np.save(fn, displacements)


@dataclass
class ExportVelocityField(Exporter):
    """Export velocity field."""
    resolution: int = 256

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        model: Model = pipeline.model
        model.eval()

        assert hasattr(model, "deformation_field")

        x = torch.linspace(-1, 1, self.resolution, device=model.device)
        X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
        pos = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
        pos_dataset = TensorDataset(pos)
        dataloader = DataLoader(pos_dataset, batch_size=1024 * 1024, shuffle=False)

        t = torch.linspace(0, 1, 11, device=model.device).view(-1, 1)
        for ti in t:
            fn = self.output_dir / f"velocity_t_{ti.item():.2f}.npz"

            velocities = []
            for batch in track(dataloader, description=f"Computing velocities at t={ti.item():.4g}"):
                pos1 = batch[0].clone()
                with torch.no_grad():
                    u = model.deformation_field.velocity(pos1[:, 0], pos1[:, 1], pos1[:, 2], ti)
                u = u.cpu().numpy()
                velocities.append(u)
            velocities = np.concatenate(velocities, axis=0).squeeze()
            velocities = velocities.reshape((self.resolution, self.resolution, self.resolution, 3))
            np.savez_compressed(fn, velocities=velocities)


@dataclass
class ExportGrowthField(Exporter):
    """Export growth rate field G(x,t) at multiple timesteps.
    
    NEW: dedicated exporter for the growth field.
    Outputs shape: (resolution, resolution, resolution) per timestep.
    """
    resolution: int = 256
    """Resolution of spatial grid."""
    num_timesteps: int = 11
    """Number of timesteps to export (evenly spaced in [0, 1])."""

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        model: Model = pipeline.model
        model.eval()

        assert hasattr(model, "deformation_field")
        deformation_field = model.deformation_field

        if not hasattr(deformation_field, 'growth_rate'):
            print("Model does not have a growth field. Nothing to export.")
            return
        # Check growth capability: shared encoder sets growth_nn=None but has growth_head
        has_growth = (
            getattr(deformation_field, '_use_shared_encoder', False)
            or (hasattr(deformation_field, 'growth_nn')
                and deformation_field.growth_nn is not None)
        )
        if not has_growth:
            print("Growth field is disabled. Nothing to export.")
            return

        x = torch.linspace(-1, 1, self.resolution, device=model.device)
        X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
        pos = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
        pos_dataset = TensorDataset(pos)
        dataloader = DataLoader(pos_dataset, batch_size=1024 * 1024, shuffle=False)

        times = torch.linspace(0, 1, self.num_timesteps, device=model.device)
        for ti in times:
            fn = self.output_dir / f"growth_rate_t_{ti.item():.2f}.npz"

            growth_rates = []
            for batch in track(dataloader, description=f"Computing growth rate at t={ti.item():.4g}"):
                p = batch[0]
                with torch.no_grad():
                    g = deformation_field.growth_rate(p[:, 0], p[:, 1], p[:, 2], ti)
                growth_rates.append(g.cpu().numpy())
            growth_rates = np.concatenate(growth_rates, axis=0)
            growth_rates = growth_rates.reshape(
                (self.resolution, self.resolution, self.resolution)
            )
            np.savez_compressed(fn, growth_rate=growth_rates)
            print(f"  t={ti.item():.2f}: mean={growth_rates.mean():.6f}, "
                  f"max={growth_rates.max():.6f}, "
                  f"nonzero={np.count_nonzero(growth_rates > 1e-6)}")


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ExportVolumeGrid, tyro.conf.subcommand(name="volume-grid")],
        Annotated[ExportImageStack, tyro.conf.subcommand(name="image-stack")],
        Annotated[ExportDeformationField, tyro.conf.subcommand(name="deformation-field")],
        Annotated[ExportVelocityField, tyro.conf.subcommand(name="velocity-field")],
        Annotated[ExportGrowthField, tyro.conf.subcommand(name="growth-field")],
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
