"""
Nerfstudio method config for dendrite growth-aware 4D reconstruction.
PARAMETER-ADJUSTED VERSION for:
  - 25 frames (1615-1663, odd frames only)
  - Segmented binary dendrite GT (not raw attenuation)
  - Synthetic parallel-beam projections at 2 orthogonal angles
  - Single canonical at last frame (frame 1663, t=1.0)

Registers 'dendrite_growth_xray' as a nerfstudio method.

Multi-resolution training pipeline:
  Stage 0: Train canonical field on last-frame 3D GT volume [separate config]
  Stage 1: Growth-aware vfield at 6^3   [this config, 3000 iter]
  Stage 2: Refine to 9^3               [refine_growth_vfield.py]
  Stage 3: Refine to 15^3              [refine_growth_vfield.py]
  Stage 4: Refine to 27^3              [refine_growth_vfield.py]
  Stage 5: (optional) Refine to 51^3   [refine_growth_vfield.py]

PARAMETER CHOICES EXPLAINED (vs original):
  - flat_field_value=0.0: synthetic projections from segmented GT have no
    background absorption; flat field is unnecessary
  - timedelta=0.05: 25 frames -> ODE needs ~20 steps for full span; 0.05
    balances accuracy vs compute (was 0.1)
  - use_gradient_checkpointing=True: 20 steps warrants checkpointing
  - growth_nn_gain=1e-4: binary GT means growth is 0->1 jump; small init
    prevents early training instability
  - growth_sparsity_coefficient=5e-3: higher than default because
    binary GT makes interface localization more critical
  - growth_negativity_coefficient=5e-2: stronger non-negativity because
    segmented data has no physical reason for density decrease
  - growth_temporal_monotonicity_start_step=200: earlier start because
    binary GT strictly satisfies monotonicity
  - volumetric_supervision_coefficient=0.01: stronger self-consistency
    because canonical is high quality (direct 3D GT, not reconstructed)
  - time_proposal_steps=300: progressive time expansion over ~300 steps
    (train near-T frames first, then expand to earlier frames)
"""
from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import (
    AdamOptimizerConfig,
    RAdamOptimizerConfig,
    AdamWOptimizerConfig,
)
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from nerf_xray.xray_temporal_datamanager import XrayTemporalDataManagerConfig
from nerf_xray.xray_dataparser import XrayDataParserConfig

# Import growth-aware modules
from .growth_deformation_fields import GrowthAwareVelocityField3dConfig
from .growth_vfield_model import GrowthVfieldModelConfig
from .growth_vfield_pipeline import GrowthVfieldPipelineConfig


dendrite_growth_xray = MethodSpecification(
    config=TrainerConfig(
        method_name="dendrite_growth_xray",
        steps_per_eval_batch=10,
        steps_per_eval_all_images=1000000,
        steps_per_eval_image=500,
        steps_per_save=5000,
        max_num_iterations=3001,
        mixed_precision=True,
        load_scheduler=False,
        load_optimizer=False,
        pipeline=GrowthVfieldPipelineConfig(
            datamanager=XrayTemporalDataManagerConfig(
                dataparser=XrayDataParserConfig(
                    auto_scale_poses=False,
                    center_method='none',
                    downscale_factors={'train': 1, 'val': 1, 'test': 1},
                    eval_mode='filename+modulo',
                    includes_time=True,
                ),
                train_num_rays_per_batch=1024,
                eval_num_rays_per_batch=2048,
                # 25 frames x 2 angles = 50 images total
                # sample up to 3 timestamps per training step
                max_images_per_timestamp=3,
                # Progressive time expansion:
                # First ~300 steps: only sample frames near T=1 (canonical)
                # This lets velocity+growth fields learn small deformations first
                # then gradually expand to earlier frames with larger deformations
                # For 25 frames this means ~12 steps per new frame introduced
                time_proposal_steps=300,
            ),
            model=GrowthVfieldModelConfig(
                use_appearance_embedding=False,
                # ==== CHANGED: white background for synthetic projections ====
                background_color='white',
                # ==== CHANGED: flat_field=0 for synthetic projections ====
                # Synthetic projections from segmented GT have no background
                # absorption, so flat field correction is unnecessary.
                # Original was 0.02 for real X-ray data.
                flat_field_value=0.0,
                flat_field_trainable=False,
                eval_num_rays_per_chunk=1024,
                num_nerf_samples_per_ray=512,
                disable_scene_contraction=True,
                # Stage 2: freeze canonical, train deformation+growth
                train_density_field=False,
                train_deformation_field=True,
                # Growth-aware velocity field
                deformation_field=GrowthAwareVelocityField3dConfig(
                    support_range=[(-1, 1), (-1, 1), (-1, 1)],
                    num_control_points=(6, 6, 6),  # coarse start
                    weight_nn_width=16,
                    weight_nn_gain=1e-3,
                    # ==== CHANGED: timedelta=0.05 (was 0.1) ====
                    # 25 frames: ODE from t=0 to t=1 needs ~20 steps at dt=0.05
                    # This gives better temporal resolution than 10 steps at 0.1
                    # Tradeoff: 2x more ODE steps -> 2x slower per iteration
                    timedelta=0.05,
                    # Growth-specific parameters
                    enable_growth=True,
                    growth_num_control_points=None,  # same as velocity
                    growth_nn_width=16,  # unused when use_shared_encoder=True; width from weight_nn_width
                    # Small init gain: binary GT means growth is a sharp 0->1 jump
                    # Start near zero, let projection loss drive activation
                    growth_nn_gain=1e-4,
                    # ==== CHANGED: checkpointing ON ====
                    # 20 ODE steps with both velocity+growth = 40-layer graph
                    # Checkpointing saves ~60% peak memory at ~30% speed cost
                    use_gradient_checkpointing=True,
                ),
                # Single canonical at last frame
                canonical_time=1.0,
                # ==== softplus: smooth activation, binary GT has strictly non-negative growth ====
                growth_activation='softplus',
            ),
            # ==========================================
            # Volumetric self-consistency at t=T
            # ==========================================
            volumetric_supervision=True,
            volumetric_supervision_start_step=100,
            # ==== CHANGED: 0.01 (was 0.005) ====
            # Stronger self-consistency because canonical is trained
            # directly from 3D GT volume (high quality, no reconstruction artifacts)
            volumetric_supervision_coefficient=0.01,

            # ==========================================
            # Growth regularization
            # ==========================================

            # ==== CHANGED: 5e-3 (was 1e-3) ====
            # Stronger sparsity for binary GT:
            # Growth should be strictly localized at solid-liquid interface
            # Binary data has sharp boundaries -> L1 should be aggressive
            growth_sparsity_coefficient=5e-3,
            growth_sparsity_start_step=0,

            # ==== CHANGED: 5e-2 (was 1e-2) ====
            # Stronger non-negativity for segmented data:
            # There is absolutely no physical reason for density decrease
            # in segmented dendrite data (no remelting, no noise fluctuation)
            growth_negativity_coefficient=5e-2,
            growth_negativity_start_step=0,

            # ==== CHANGED: start at 200 (was 500) ====
            # Binary GT strictly satisfies monotonicity (solid only grows)
            # Can enforce this earlier since the constraint is exact
            growth_temporal_monotonicity_coefficient=1e-3,
            growth_temporal_monotonicity_start_step=200,
            growth_temporal_monotonicity_every_n_steps=5,

            # Directional: set >0 if your solidification has clear z-direction
            # Your data appears to grow roughly uniformly -> keep at 0
            growth_directional_coefficient=0.0,

            # ==== Flat field loss: 0 since flat_field is fixed at 0 ====
            flat_field_loss_multiplier=0.0,
        ),
        # Optimizer groups must match GrowthVfieldModel.get_param_groups():
        #   "proposal_networks", "fields", "flat_field", "growth_field"
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.0001, max_steps=200000
                ),
            },
            "fields": {
                "optimizer": AdamWOptimizerConfig(
                    lr=1e-3, eps=1e-15, weight_decay=1e-8
                ),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=3e-4, max_steps=10000
                ),
            },
            "flat_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-6, max_steps=50000
                ),
            },
            "growth_field": {
                "optimizer": AdamWOptimizerConfig(
                    lr=1e-3, eps=1e-15, weight_decay=0.0  # 无 weight_decay！
                ),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=3000
                ),
            },
        },
        viewer=ViewerConfig(
            num_rays_per_chunk=1 << 15,
            camera_frustum_scale=0.5,
            quit_on_train_completion=True,
        ),
        vis="tensorboard",
    ),
    description="Dendrite growth-aware 4D X-ray CT reconstruction (25-frame segmented GT).",
)
