"""
Nerfstudio Template Pipeline
"""

import typing
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Type, Union

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from jaxtyping import Float, Shaped
from nerfstudio.data.datamanagers.base_datamanager import (DataManager,
                                                           DataManagerConfig,
                                                           VanillaDataManager)
from nerfstudio.data.datamanagers.full_images_datamanager import \
    FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import \
    ParallelDataManager
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (VanillaPipeline,
                                                VanillaPipelineConfig)
from nerfstudio.utils import profiler
from nerfstudio.utils.rich_utils import CONSOLE
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nerf_xray.xray_datamanager import XrayDataManagerConfig
from nerf_xray.canonical_model import CanonicalModel, CanonicalModelConfig



@dataclass
class CanonicalPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: CanonicalPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = field(default_factory=lambda: XrayDataManagerConfig)
    """specifies the datamanager config"""
    model: ModelConfig = field(default_factory=lambda: CanonicalModelConfig)
    """specifies the model config"""
    volumetric_supervision: bool = False
    """specifies if the training gets volumetric supervision"""
    volumetric_supervision_start_step: int = 100
    """start providing volumetric supervision at this step"""
    volumetric_supervision_coefficient: float = 0.005
    """coefficient for the volumetric supervision loss"""
    load_density_ckpt: Optional[Path] = None
    """specifies the path to the density field to load"""
    flat_field_penalty: float = 0.01
    """penalty to increase flat field"""


class CanonicalPipeline(VanillaPipeline):
    """Canonical Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
        self,
        config: CanonicalPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)

        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
        )
        self.model.to(device)

        if config.load_density_ckpt is not None:
            self.load_density_field(config.load_density_ckpt)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                CanonicalModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])

    @profiler.time_function
    def get_eval_loss_dict(self, step: int) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        assert batch['image'].ndim==2
        batch['image'] = batch['image'][...,[0]] # [..., 1]
        model_outputs = self.model(ray_bundle)
        metrics_dict: Dict
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        if self.datamanager.object is not None:
            metrics_dict.update(self.calculate_density_loss())
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict
    
    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        assert isinstance(self.datamanager, (VanillaDataManager, ParallelDataManager, FullImageDatamanager))
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for camera, batch in self.datamanager.fixed_indices_eval_dataloader:
                # time this the following line
                inner_start = time()
                outputs = self.model.get_outputs_for_camera(camera=camera)
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)
                if output_path is not None:
                    raise NotImplementedError("Saving images is not implemented yet")

                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )
        if self.datamanager.object is not None:
            # evaluate volumetric loss on a 100x100x100 grid
            metrics_dict.update(self.calculate_density_loss())
        self.train()
        return metrics_dict
    
    def calculate_density_loss(self, sampling: str = 'random', resolution: Optional[int] = None) -> Dict[str, Any]:
        if sampling=='grid':
            if resolution is None:
                resolution = 250
            pos = torch.linspace(-1, 1, resolution, device=self.device) # scene box goes between -1 and 1 
            pos = torch.stack(torch.meshgrid(pos, pos, pos, indexing='ij'), dim=-1).reshape(-1, 3)
        elif sampling=='random':
            if resolution is None:
                resolution = self.config.datamanager.train_num_rays_per_batch*32
            pos = 2*torch.rand((resolution, 3), device=self.device) - 1.0
        
        object = self.datamanager.object
        pred_density = self._model.field.get_density_from_pos(pos).squeeze()

        density = object.density(pos).squeeze() # density between -1 and 1
        
        x = density
        y = pred_density

        density_loss = torch.nn.functional.mse_loss(y, x)

        density_n = (x - x.min()) / (x.max() - x.min())
        pred_dens_n = (y - y.min()) / (y.max() - y.min())
        scaled_density_loss = torch.nn.functional.mse_loss(pred_dens_n, density_n)
        
        mux = x.mean()
        muy = y.mean()
        dx = x-mux
        dy = y-muy
        normed_correlation = torch.sum(dx*dy) / torch.sqrt(dx.pow(2).sum() * dy.pow(2).sum())
        return {
            'volumetric_loss': density_loss, 
            'scaled_volumetric_loss': scaled_density_loss,
            'normed_correlation': normed_correlation
            }

    def get_flat_field_penalty(self):
        return -self.config.flat_field_penalty*self.model.flat_field

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step)
        assert batch['image'].ndim==2
        batch['image'] = batch['image'][...,[0]] # [..., 1]
        model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        loss_dict['flat_field_loss'] = self.get_flat_field_penalty()
        
        if self.config.volumetric_supervision and step>self.config.volumetric_supervision_start_step:
            # provide supervision to visual training. Use cross-corelation loss
            assert self.datamanager.object is not None
            density_loss = self.calculate_density_loss(sampling='random')
            loss_dict['volumetric_loss'] = self.config.volumetric_supervision_coefficient*(1-density_loss['normed_correlation'])

        return model_outputs, loss_dict, metrics_dict
    
    def eval_along_plane(
        self, 
        target: Literal['field', 'datamanager', 'both'],
        plane='xy', 
        distance=0.0, 
        fn=None, 
        engine='cv', 
        resolution=500,
        rhomax=1.0,
        time=0.0,
        which=None
    ):
        a = torch.linspace(-1, 1, resolution, device=self.device) # scene box will map to 0-1
        b = torch.linspace(-1, 1, resolution, device=self.device) # scene box will map to 0-1
        A,B = torch.meshgrid(a,b, indexing='ij')
        C = distance*torch.ones_like(A)
        if plane == 'xy':
            pos = torch.stack([A, B, C], dim=-1)
        elif plane == 'yz':
            pos = torch.stack([C, A, B], dim=-1)
        elif plane == 'xz':
            pos = torch.stack([A, C, B], dim=-1)
        if target in ['field', 'both']:
            with torch.no_grad():
                if self.model.deformation_field is not None:
                    pred_density = self._model.field.get_density_from_pos(pos, deformation_field=self._model.deformation_field, time=time).squeeze()
                else:
                    pred_density = self._model.field.get_density_from_pos(pos).squeeze()
                pred_density = pred_density.cpu().numpy() / rhomax
        if target in ['datamanager', 'both']:
            pos_shape = pos.shape
            assert pos_shape[-1] == 3
            obj_density = self.datamanager.object.density(pos.view(-1,3)).view(pos_shape[:-1])
            max_density = self.datamanager.object.max_density
            obj_density = obj_density.cpu().numpy() / max_density
        if target == 'both':
            density = np.concatenate([obj_density, pred_density], axis=1)
        elif target == 'field':
            density = pred_density
        elif target == 'datamanager':
            density = obj_density

        if engine=='matplotlib':
            plt.figure(figsize=(6,6) if target!='both' else (12,6))
            plt.imshow(
                density, 
                extent=[-1,1,-1,1] if plane=='xy' else [-1,3,-1,1], 
                origin='lower', cmap='gray', vmin=0, vmax=1
            )
            if fn is not None:
                plt.savefig(fn)
            plt.close()
        elif engine in ['cv', 'opencv']:
            density = np.clip(density, 0, 1)
            density = (density*255).astype(np.uint8)
            if fn is not None:
                if isinstance(fn, Path):
                    fn = fn.as_posix()
                cv.imwrite(fn, density)
        elif engine=='numpy':
            return density
        else:
            raise ValueError(f"Invalid engine {engine}")