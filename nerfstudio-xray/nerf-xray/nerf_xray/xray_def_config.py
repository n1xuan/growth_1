"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import \
    NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import (AdamOptimizerConfig,
                                          RAdamOptimizerConfig,
                                          AdamWOptimizerConfig)
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from nerf_xray.xray_temporal_datamanager import XrayTemporalDataManagerConfig
from nerf_xray.xray_dataparser import XrayDataParserConfig
from nerf_xray.canonical_model import CanonicalModelConfig
from nerf_xray.canonical_pipeline import CanonicalPipelineConfig
from nerf_xray.deformation_fields import (BsplineTemporalDeformationField3dConfig,
                                          BsplineDeformationField3dConfig)

nerf_def_xray = MethodSpecification(
    config=TrainerConfig(
        method_name="nerf_def_xray", 
        steps_per_eval_batch=10,
        steps_per_eval_all_images=1000000,
        steps_per_eval_image=500,
        steps_per_save=5000,
        max_num_iterations=501,
        mixed_precision=True,
        load_scheduler=False,
        load_optimizer=False,
        pipeline=CanonicalPipelineConfig(
            datamanager=XrayTemporalDataManagerConfig(
                dataparser=XrayDataParserConfig(
                    auto_scale_poses=False,
                    center_method='none',
                    orientation_method='none',
                    downscale_factors={'train': 1, 'val': 8, 'test': 8},
                    eval_mode='filename+modulo',
                    includes_time=True,
                ),
                train_num_rays_per_batch=1024,
                eval_num_rays_per_batch=2048,
                max_images_per_timestamp=3,
                time_proposal_steps=500,
            ),
            model=CanonicalModelConfig(
                use_appearance_embedding=False,
                background_color='white',
                flat_field_value=0.00,
                flat_field_trainable=True,
                eval_num_rays_per_chunk=1024,
                num_nerf_samples_per_ray=512,
                disable_scene_contraction=True,
                train_density_field=False,
                train_deformation_field=True,
                deformation_field=BsplineTemporalDeformationField3dConfig(
                    support_range=[(-1,1),(-1,1),(-1,1)],
                    num_control_points=(4,4,4),
                )
            ),
            volumetric_supervision=False,
        ),
        optimizers={
            # TODO: consider changing optimizers depending on your custom method
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamWOptimizerConfig(lr=1e-3, eps=1e-15, weight_decay=1e-8),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=3e-4, max_steps=10000),
            },
            # "fields": {
            #     "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
            #     "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            # },
            "flat_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=50000),
            },
            "camera_opt": {
                # "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-15),
                # "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=5000),
                "optimizer": AdamOptimizerConfig(lr=1e-11, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-12, max_steps=5000),
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
