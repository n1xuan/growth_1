"""
Nerfstudio Template Pipeline
"""

import typing
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Type, Union, List

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
from nerf_xray.vfield_model import VfieldModel, VfieldModelConfig
from .objects import Object


@dataclass
class VfieldPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: VfieldPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = field(default_factory=lambda: XrayDataManagerConfig)
    """specifies the datamanager config"""
    model: ModelConfig = field(default_factory=lambda: VfieldModelConfig)
    """specifies the model config"""
    volumetric_supervision: bool = False
    """specifies if the training gets volumetric supervision"""
    volumetric_supervision_start_step: int = 100
    """start providing volumetric supervision at this step"""
    volumetric_supervision_coefficient: float = 0.005
    """coefficient for the volumetric supervision loss"""
    density_mismatch_start_step: int = -1
    """start providing displacement field closure at this step"""
    density_mismatch_coefficient: float = 1e-3
    """multiplicative factor for field closure"""
    load_density_ckpt: Optional[Path] = None
    """specifies the path to the density field to load"""
    flat_field_loss_multiplier: float = 0.001
    """multiplier for flat field regularization"""


class VfieldPipeline(VanillaPipeline):
    """Template Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
        self,
        config: VfieldPipelineConfig,
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

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                VfieldModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        loss_dict['flat_field_loss'] = self.get_flat_field_penalty()

        if self.config.density_mismatch_start_step>=0 and step>self.config.density_mismatch_start_step and (self.model.field_f is not None) and (self.model.field_b is not None):
            loss_dict.update({'mismatch_penalty':self.config.density_mismatch_coefficient*self.get_fields_mismatch_penalty()})
        
        
        if self.config.volumetric_supervision and step>self.config.volumetric_supervision_start_step:
            # provide supervision to visual training. Use cross-corelation loss
            density_loss = self.calculate_density_loss(sampling='random')
            loss_dict[f'volumetric_loss_0'] = self.config.volumetric_supervision_coefficient*(1-density_loss['normed_correlation'])
            density_loss = self.calculate_density_loss(sampling='random', time=1.0)
            loss_dict[f'volumetric_loss_1'] = self.config.volumetric_supervision_coefficient*(1-density_loss['normed_correlation'])

        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_loss_dict(self, step: int) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict: Dict
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        metrics_dict.update(
            {'flat_field': self.model.flat_field.phi_x.mean()}
        )
        metrics_dict['normed_correlation_0'] = self.calculate_density_loss(sampling='random', time=0.0)['normed_correlation']
        metrics_dict['normed_correlation_1'] = self.calculate_density_loss(sampling='random', time=1.0)['normed_correlation']
        with torch.no_grad():
            metrics_dict.update({'mismatch_penalty':self.get_fields_mismatch_penalty()})
        if self.model.config.disable_mixing==False:
            metrics_dict.update(self.model.field_weighing.get_stat_dict())

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict
    
    @profiler.time_function
    def get_average_eval_image_metrics(
        self, 
        step: Optional[int] = None, 
        output_path: Optional[Path] = None, 
        get_std: bool = False, 
        which: Optional[Literal['forward','backward','mixed']] = None
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.
            which: which field to evaluate

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
                outputs = self.model.get_outputs_for_camera(camera=camera, which=which)
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)
                loss_dict = self.model.get_loss_dict(outputs, batch, metrics_dict)
                # convert any tensors to floats
                for key in loss_dict.keys():
                    if isinstance(loss_dict[key], torch.Tensor):
                        loss_dict[key] = loss_dict[key].item()
                        
                if output_path is not None:
                    raise NotImplementedError("Saving images is not implemented yet")

                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
                # Save image info to metrics dict
                image_idx = batch['image_idx']
                img_filename = self.datamanager.eval_dataset.image_filenames[image_idx]
                metrics_dict["image_name"] = img_filename.as_posix()
                metrics_dict["image_time"] = camera.times.item()
                metrics_dict.update(loss_dict) # try sticking it into metrics dict
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if isinstance(metrics_dict_list[0][key], str):
                continue # cannot average strings
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
        # Store non-averaged metrics list
        metrics_dict['metrics_list'] = metrics_dict_list

        self.train()
        return metrics_dict

    def get_eval_density_loss(
        self, 
        sampling: Literal['random','grid'] = 'random', 
        time: Optional[float] = None, 
        target: Optional[Object] = None, 
        npoints: Optional[int] = None,
        extent: Optional[Tuple[Tuple[float,float],Tuple[float,float],Tuple[float,float]]] = field(default_factory=lambda: ((-1,1),(-1,1),(-1,1))),
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        if sampling=='grid':
            if npoints is None: 
                npoints = 200
            lsp = torch.linspace(0, 1, npoints, device=self.device) # scene box goes between -1 and 1 
            pos = torch.stack(
                torch.meshgrid(
                    extent[0][0] + lsp*(extent[0][1]-extent[0][0]), 
                    extent[1][0] + lsp*(extent[1][1]-extent[1][0]), 
                    extent[2][0] + lsp*(extent[2][1]-extent[2][0]), 
                    indexing='ij'
                ), 
                dim=-1
            ).reshape(-1, 3)
        elif sampling=='random':
            if npoints is None:
                npoints = self.config.datamanager.train_num_rays_per_batch*32
            pos = torch.rand((npoints, 3), device=self.device)
            pos[:,0] = extent[0][0] + pos[:,0]*(extent[0][1]-extent[0][0])
            pos[:,1] = extent[1][0] + pos[:,1]*(extent[1][1]-extent[1][0])
            pos[:,2] = extent[2][0] + pos[:,2]*(extent[2][1]-extent[2][0])

        # need to implement batching as it won't necessarily fit into memory
        if batch_size is None:
            batch_size = 1<<50
        pred_density_f = []
        pred_density_b = []
        mixed_density = []
        num_batches = 0
        for i in range(0, pos.shape[0], batch_size):
            pos_batch = pos[i:i+batch_size]
            _pred_density_f = self.model.field_f.get_density_from_pos(pos_batch, deformation_field=lambda x,t: self.model.deformation_field(x,t,0.0), time=time).squeeze()
            _pred_density_b = self.model.field_b.get_density_from_pos(pos_batch, deformation_field=lambda x,t: self.model.deformation_field(x,t,1.0), time=time).squeeze()
            _mixed_density = self.model.get_density_from_pos(pos_batch, time=time, which='mixed').squeeze()
            pred_density_f.append(_pred_density_f)
            pred_density_b.append(_pred_density_b)
            mixed_density.append(_mixed_density)
            num_batches += 1
        pred_density_f = torch.cat(pred_density_f, dim=0)
        pred_density_b = torch.cat(pred_density_b, dim=0)
        mixed_density = torch.cat(mixed_density, dim=0)

        if target is None:
            assert time in [0.0, 1.0], "Time must be 0.0 or 1.0"
            obj = self.datamanager.object if time==0.0 else self.datamanager.final_object
        else:
            obj = target

        density = obj.density(pos).squeeze() # density between -1 and 1
        
        normed_correlation_f = self.calculate_normed_correlation(x=density, y=pred_density_f)
        normed_correlation_b = self.calculate_normed_correlation(x=density, y=pred_density_b)
        normed_correlation_mixed = self.calculate_normed_correlation(x=density, y=mixed_density)
        mismatch_penalty = (pred_density_f - pred_density_b).pow(2).mean()
        normalized_mismatch_penalty = mismatch_penalty / mixed_density.pow(2).mean()
        mismatch_correlation = self.calculate_normed_correlation(x=pred_density_f, y=pred_density_b)
        return {
            'normed_correlation_f': normed_correlation_f,
            'normed_correlation_b': normed_correlation_b,
            'normed_correlation_mixed': normed_correlation_mixed,
            'self_mismatch_penalty': mismatch_penalty,
            'normalized_mismatch_penalty': normalized_mismatch_penalty,
            'mismatch_correlation': mismatch_correlation,
            }
    
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
        which: Optional[Literal['forward','backward','mixed']] = None
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
                pred_density = self._model.get_density_from_pos(pos, time=time, which=which).squeeze()
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

    def calculate_density_loss(self, sampling: str = 'random', time: float = 0.0) -> Dict[str, Any]:
        if sampling=='grid':
            pos = torch.linspace(-0.7, 0.7, 200, device=self.device) # scene box goes between -1 and 1 
            pos = torch.stack(torch.meshgrid(pos, pos, pos, indexing='ij'), dim=-1).reshape(-1, 3)
        elif sampling=='random':
            pos = (torch.rand((self.config.datamanager.train_num_rays_per_batch*32, 3), device=self.device) - 0.5) * 1.4
        if time==0.0:
            density_0 = self.model.get_density_from_pos(pos, time=0.0, which='backward').squeeze() # points sampled at 0.0
            density_1 = self.model.get_density_from_pos(pos, time=0.0, which='forward').squeeze()
        elif time==1.0:
            density_0 = self.model.get_density_from_pos(pos, time=1.0, which='forward').squeeze()
            density_1 = self.model.get_density_from_pos(pos, time=1.0, which='backward').squeeze()
        else:
            raise ValueError(f'Time {time} not supported')
        
        normed_correlation = self.calculate_normed_correlation(x=density_0, y=density_1)
        return {
            'normed_correlation': normed_correlation
            }
    
    @staticmethod
    def calculate_normed_correlation(x, y):
        mux = x.mean()
        muy = y.mean()
        dx = x-mux
        dy = y-muy
        normed_correlation = torch.sum(dx*dy) / torch.sqrt(dx.pow(2).sum() * dy.pow(2).sum())
        return normed_correlation

    def get_flat_field_penalty(self):
        return -self.config.flat_field_loss_multiplier*self.model.flat_field.phi_x.mean()

    def get_fields_mismatch_penalty(
            self, 
            reduction: Literal['mean','sum','none'] = 'mean', 
            npoints: Optional[int] = None,
            extent: Optional[float] = 1.0
    ):
        if self.model.config.direction != 'both':
            return torch.zeros(1, device=self.device)
        # sample time
        times = torch.linspace(0, 1, 11, device=self.device) # 0 to 1
        if self.training: # perturb
            times += (torch.rand(11)-0.5)*0.1
        diffs = []
        if npoints is None:
            npoints = self.config.datamanager.train_num_rays_per_batch*32
        '''
        for i,t in enumerate(times):
            pos = (2*torch.rand((npoints, 3), device=self.device) - 1.0) * extent
            diff = self.model.get_density_difference(pos, t.item()).pow(2).mean().view(1)
            diffs.append(diff)
        '''
        chunk_size = 1024
        for i,t in enumerate(times):
            pos = (2*torch.rand((npoints, 3), device=self.device) - 1.0) * extent
            diff_accum = []
            for ci in range(0, pos.shape[0], chunk_size):
                pos_chunk = pos[ci:ci+chunk_size]
                diff_chunk = self.model.get_density_difference(pos_chunk, t.item()).pow(2).mean().view(1)
                diff_accum.append(diff_chunk * pos_chunk.shape[0])
            diff = (torch.stack(diff_accum).sum() / pos.shape[0]).view(1)
            diffs.append(diff)
        if len(diffs)>0:
            if reduction=='sum':
                loss = torch.cat(diffs).sum()
            elif reduction=='mean':
                loss = torch.cat(diffs).mean()
            elif reduction=='none':
                loss = torch.cat(diffs)
            else:
                raise ValueError(f'`reduction` {reduction} not recognized')
        else:
            loss = t.new_zeros(1)
        return loss