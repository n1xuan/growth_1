"""
Configuration for the canonical volume.
"""

from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import \
    NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import (AdamOptimizerConfig,
                                          RAdamOptimizerConfig)
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from nerf_xray.xray_datamanager import XrayDataManagerConfig
from nerf_xray.multi_camera_dataparser import MultiCameraDataParserConfig
from nerf_xray.xray_dataparser import XrayDataParserConfig
from nerf_xray.canonical_model import CanonicalModelConfig
from nerf_xray.canonical_pipeline import CanonicalPipelineConfig
from nerf_xray.utils import ColdRestartLinearDecaySchedulerConfig


nerf_xray = MethodSpecification(
    config=TrainerConfig(
        method_name="nerf_xray", 
        steps_per_eval_batch=10,
        steps_per_eval_all_images=100000,
        steps_per_eval_image=500,
        steps_per_save=5000,
        max_num_iterations=1001,
        mixed_precision=True,
        pipeline=CanonicalPipelineConfig(
            datamanager=XrayDataManagerConfig(
                dataparser=MultiCameraDataParserConfig(
                    auto_scale_poses=False,
                    center_method='none',
                    downscale_factors={'train': 1, 'val': 8, 'test': 8},
                    eval_mode='filename+modulo',
                ),
                train_num_rays_per_batch=1024,
                eval_num_rays_per_batch=2048,
            ),
            model=CanonicalModelConfig(
                use_appearance_embedding=False,
                background_color='white',
                flat_field_value=0.0,
                flat_field_trainable=True,
                eval_num_rays_per_chunk=1024,
                num_nerf_samples_per_ray=512,
                disable_scene_contraction=True,
                interlevel_loss_mult=0.0,
                distortion_loss_mult=0.0,
            ),
            volumetric_supervision=False,
        ),
        optimizers={
            # TODO: consider changing optimizers depending on your custom method
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=5000),
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ColdRestartLinearDecaySchedulerConfig(warmup_steps=50, lr_final=3e-5, max_steps=5000),
                # "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=5000),
            },
            "flat_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ColdRestartLinearDecaySchedulerConfig(warmup_steps=200, lr_final=1e-6, max_steps=5000),
            },
        },
        viewer=ViewerConfig(
            num_rays_per_chunk=1 << 15, 
            camera_frustum_scale=0.5,
            quit_on_train_completion=True,
        ),
        vis="tensorboard",
    ),
    description="Nerfstudio method template.",
)
