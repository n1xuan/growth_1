"""
Growth-aware velocity field model for dendrite 4D reconstruction.
CORRECTED VERSION — includes _apply_growth_activation, PlaceHolderField aliases, dtype fix.

Key changes from VfieldModel:
1. Single canonical field (last frame, t=T) instead of dual forward/backward
2. Deformation field returns (warped_pos, growth_accum) tuple12
3. Growth correction added to canonical density via closure pattern
4. No field_weighing mixing
5. direction='backward' only (ODE integrates from query t to t_ref=1.0)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type, Union, Optional
from math import ceil
from contextlib import contextmanager
from collections import defaultdict

import numpy as np
import torch
from jaxtyping import Float, Shaped
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import (TrainingCallback,
                                         TrainingCallbackAttributes,
                                         TrainingCallbackLocation)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.losses import (
    MSELoss, interlevel_loss, orientation_loss,
    pred_normal_loss, scale_gradients_by_distance_squared)
from nerfstudio.model_components.ray_samplers import (ProposalNetworkSampler,
                                                      UniformSampler)
from nerfstudio.model_components.renderers import (AccumulationRenderer,
                                                   DepthRenderer,
                                                   NormalsRenderer)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.data.scene_box import OrientedBox, SceneBox
from torch import Tensor
from torch.nn import Parameter

from nerf_xray.deformation_fields import (
    DeformationFieldConfig,
    IdentityDeformationField,
    BSplineField1d,
)
from nerf_xray.canonical_field import CanonicalNerfField, PlaceHolderField
from nerf_xray.xray_renderer import AttenuationRenderer


@dataclass
class GrowthVfieldModelConfig(NerfactoModelConfig):
    """Configuration for growth-aware velocity field model."""
    _target: Type = field(default_factory=lambda: GrowthVfieldModel)

    train_density_field: bool = True
    train_deformation_field: bool = False
    deformation_field: DeformationFieldConfig = field(
        default_factory=lambda: DeformationFieldConfig
    )
    flat_field_value: float = 0.0
    flat_field_trainable: bool = False
    camera_optimizer = None
    growth_activation: Literal['relu', 'softplus', 'none'] = 'relu'
    """Activation applied to growth_accum before adding to density."""
    canonical_time: float = 1.0
    """Time of the canonical (reference) frame. 1.0 = last frame."""


class GrowthVfieldModel(Model):
    """Single-canonical growth-aware velocity field model.
    
    Density query at (x, t):
        warped_pos, growth_accum = deformation_field(x, t, t_ref=T)
        rho_canonical = canonical_field(warped_pos)
        rho(x, t) = rho_canonical - activation(growth_accum)
    """

    config: GrowthVfieldModelConfig

    def populate_modules(self):
        """Set up fields and modules."""
        super().populate_modules()

        assert self.config.disable_scene_contraction, \
            "Scene contraction not supported for X-ray reconstruction."

        appearance_embedding_dim = (
            self.config.appearance_embed_dim
            if self.config.use_appearance_embedding else 0
        )

        # Single canonical field (last frame)
        self.field = CanonicalNerfField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=None,  # no scene contraction for X-ray
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            average_init_density=self.config.average_init_density,
            implementation=self.config.implementation,
        )

        # ========== BACKWARD COMPATIBILITY ALIASES ==========
        # VfieldModel has field_f + field_b; checkpoint loading may reference them.
        # field_f is unused (PlaceHolder), field_b aliases to self.field.
        self.field_f = PlaceHolderField()
        self.field_b = self.field
        # ====================================================

        # Growth-aware deformation field
        self.deformation_field = self.config.deformation_field.setup()

        # Learnable scale factor for growth magnitude 6.7
        # Initialized to 0.1 to prevent over-subtraction at training start
        # Optimizer will increase it to match canonical density scale
        self.growth_scale = torch.nn.Parameter(torch.tensor(0.1))

        # Freeze/unfreeze based on training stage
        if not self.config.train_density_field:
            self.field.requires_grad_(False)
        if not self.config.train_deformation_field:
            self.deformation_field.requires_grad_(False)

        self.camera_optimizer = None
        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations

        # Build proposal networks (unchanged from baseline)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=None,
                **prop_net_args,
                average_init_density=self.config.average_init_density,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend(
                [network.density_fn for _ in range(num_prop_nets)]
            )
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[
                    min(i, len(self.config.proposal_net_args_list) - 1)
                ]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=None,
                    **prop_net_args,
                    average_init_density=self.config.average_init_density,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend(
                [network.density_fn for network in self.proposal_networks]
            )

        # Sampler
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup],
                          [0, self.config.proposal_update_every]),
                1, self.config.proposal_update_every,
            )

        initial_sampler = None
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(
                single_jitter=self.config.use_single_jitter
            )

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(
            near_plane=self.config.near_plane,
            far_plane=self.config.far_plane,
        )

        # Renderers
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="median")
        self.renderer_expected_depth = DepthRenderer(method="expected")
        self.renderer_normals = NormalsRenderer()
        self.renderer_attenuation = AttenuationRenderer(
            background_color=self.config.background_color,
        )
        self.renderer_rgb = self.renderer_attenuation

        # Flat field
        ff = self.kwargs['metadata'].get('flat_field', None)
        if ff is not None:
            _ff = -np.log(ff)
            CONSOLE.print(f"Using flat field from metadata: {ff:.3f} -> {_ff:.3f}")
            ff = _ff
        else:
            ff = self.config.flat_field_value
        self.flat_field = BSplineField1d(
            torch.nn.parameter.Parameter(ff * torch.ones(10)),
            support_outside=True,
            support_range=(0, 1),
        )
        if not self.config.flat_field_trainable:
            self.flat_field.requires_grad_(False)

        # Shaders, losses, metrics
        self.normals_shader = NormalsShader()
        self.rgb_loss = MSELoss()
        self.step = 0

        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    # =====================================================================
    # Growth activation (CRITICAL FIX: was missing in original code)
    # =====================================================================
    def _apply_growth_activation(self, growth_accum: Tensor) -> Tensor:
        """Apply activation to raw growth accumulation.
        
        Controls physical interpretation:
        - 'relu': Non-negative only (irreversible solidification, no remelting)
        - 'softplus': Smooth non-negative (avoids zero-gradient dead zone at G=0)
        - 'none': Bidirectional (allows remelting)
        """
        if self.config.growth_activation == 'relu':
            return torch.relu(growth_accum)
        elif self.config.growth_activation == 'softplus':
            return torch.nn.functional.softplus(growth_accum, beta=5.0)
        elif self.config.growth_activation == 'none':
            return growth_accum
        else:
            raise ValueError(
                f"Unknown growth activation: {self.config.growth_activation}"
            )

    # =====================================================================
    # Deformation + growth wrapper
    # =====================================================================
    def _deformation_with_growth(
        self, positions: Tensor, times: Tensor
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Call deformation field and unpack growth_accum.
        
        Returns:
            warped_positions: [N, 3]
            growth_accum: [N] or None if growth is disabled
        """
        t_ref = self.config.canonical_time
        result = self.deformation_field(positions, times, t_ref)
        if isinstance(result, tuple):
            return result[0], result[1]
        else:
            return result, None

    # =====================================================================
    # Core forward pass
    # =====================================================================
    def get_outputs(self, ray_bundle: RayBundle, **kwargs):
        """Compute outputs for a ray bundle.
        
        Uses closure pattern to pass growth_accum from deformation field
        to density output without modifying canonical_field.py.
        """
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fns=self.density_fns
        )

        # Closure storage for growth accumulation
        growth_storage = {}

        def deformation_callback(positions, times):
            """Called by CanonicalNerfField.get_density().
            Returns warped positions; stores growth_accum in closure."""
            warped_pos, growth_accum = self._deformation_with_growth(
                positions, times
            )
            growth_storage['accum'] = growth_accum
            return warped_pos

        # Forward through canonical field
        field_outputs = self.field.forward(
            ray_samples,
            compute_normals=self.config.predict_normals,
            deformation_field=deformation_callback,
        )

        # Subtract growth correction from density (CRITICAL: dtype casting)
        if 'accum' in growth_storage and growth_storage['accum'] is not None:
            growth_accum = growth_storage['accum']
            growth_correction = self.growth_scale * self._apply_growth_activation(growth_accum)
            # Cast to match density dtype (important for mixed precision)
            growth_correction = growth_correction.to(
                field_outputs[FieldHeadNames.DENSITY].dtype
            )
            field_outputs[FieldHeadNames.DENSITY] = (
                field_outputs[FieldHeadNames.DENSITY]
                - growth_correction.unsqueeze(-1)
            )
            # Ensure non-negative total density
            field_outputs[FieldHeadNames.DENSITY] = torch.relu(
                field_outputs[FieldHeadNames.DENSITY]
            )

        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(
                field_outputs, ray_samples
            )

        weights = ray_samples.get_weights(
            field_outputs[FieldHeadNames.DENSITY]
        )
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        with torch.no_grad():
            depth = self.renderer_depth(
                weights=weights, ray_samples=ray_samples
            )
        expected_depth = self.renderer_expected_depth(
            weights=weights, ray_samples=ray_samples
        )
        accumulation = self.renderer_accumulation(weights=weights)
        attenuation = self.renderer_attenuation(
            densities=field_outputs[FieldHeadNames.DENSITY],
            ray_samples=ray_samples,
        )
        flat_field = self.flat_field(ray_bundle.times.view(-1)).view(-1, 1)
        rgb = (
            self.renderer_attenuation.merge_flat_field(attenuation, flat_field)
            * attenuation.new_ones(1, 3)
        )

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
            "attenuation": attenuation,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(
                normals=field_outputs[FieldHeadNames.NORMALS], weights=weights
            )
            pred_normals = self.renderer_normals(
                field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights
            )
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)

        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS],
                ray_bundle.directions,
            )
            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(
                weights=weights_list[i], ray_samples=ray_samples_list[i]
            )
        return outputs

    # =====================================================================
    # Density query (for volumetric supervision and export)
    # =====================================================================
    @contextmanager
    def empty_context_manager(self):
        yield

    def get_density_from_pos(
        self,
        positions: Tensor,
        time: Optional[Union[Tensor, float]] = None,
        which: Optional[str] = None,  # Ignored; kept for API compat
    ) -> Tensor:
        """Get density at arbitrary positions and time.
        rho(x,t) = rho_canonical(warp(x, t->T)) - activation(growth_accum)
        """
        if time is None:
            time = positions.new_zeros(*positions.shape[:-1], 1)
        elif isinstance(time, float):
            time = positions.new_ones(*positions.shape[:-1], 1) * time

        if self.training:
            cm = self.empty_context_manager
        else:
            cm = torch.no_grad

        with cm():
            t_ref = self.config.canonical_time
            result = self.deformation_field(positions, time, t_ref)
            
            if isinstance(result, tuple):
                warped_pos, growth_accum = result
            else:
                warped_pos, growth_accum = result, None

            density = self.field.get_density_from_pos(
                warped_pos,
                deformation_field=None,  # already warped
                time=time,
            )
            if density is not None:
                density = density.squeeze()
            else:
                return positions.new_zeros(positions.shape[:-1])

            if growth_accum is not None:
                growth_correction = self.growth_scale * self._apply_growth_activation(growth_accum)
                growth_correction = growth_correction.to(density.dtype)
                density = density - growth_correction
                density = torch.relu(density)

        return density

    # =====================================================================
    # Parameter groups
    # =====================================================================
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(
            self.proposal_networks.parameters()
        )
        
        # Canonical field + velocity (shared group)
        if self.config.train_density_field:
            param_groups["fields"] = list(self.field.parameters())
        else:
            param_groups["fields"] = []  # canonical冻结，不放进optimizer
        
        # Separate velocity and growth parameters
        if hasattr(self.deformation_field, 'growth_nn') and self.deformation_field.growth_nn is not None:
            # Velocity parameters only
            velocity_params = [p for n, p in self.deformation_field.named_parameters() 
                          if 'growth' not in n]
            param_groups["fields"].extend(velocity_params)
        
            # Growth parameters in separate group
            growth_params = [p for n, p in self.deformation_field.named_parameters() 
                        if 'growth' in n]
            growth_params.append(self.growth_scale)
            param_groups["growth_field"] = growth_params
        else:
            param_groups["fields"].extend(
                list(self.deformation_field.parameters())
            )
            # 空列表会报错，放一个dummy参数
            if not hasattr(self, '_growth_dummy'):
                self._growth_dummy = Parameter(torch.zeros(1))
            param_groups["growth_field"] = [self._growth_dummy]
    
        param_groups["flat_field"] = list(self.flat_field.parameters())
        return param_groups

    # =====================================================================
    # Callbacks
    # =====================================================================
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                self.step = step
                train_frac = np.clip(step / N, 0, 1)

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    # =====================================================================
    # Forward / multi-camera
    # =====================================================================
    def forward(
        self,
        ray_bundle: Union[List[RayBundle], RayBundle],
        which: Optional[str] = None,
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """Forward pass. Handles list of ray bundles for multi-camera."""
        if isinstance(ray_bundle, list):
            outputs = [self.forward(rb, which=which) for rb in ray_bundle]
            weights = torch.cat(
                [rb.metadata['camera_weights'] for rb in ray_bundle], dim=1
            )
            weights = weights.permute(1, 0)
            outputs = self._sum_outputs_with_weights(outputs, weights)
            return outputs
        else:
            if self.collider is not None:
                ray_bundle = self.collider(ray_bundle)
            return self.get_outputs(ray_bundle)

    def _sum_outputs_with_weights(
        self,
        outputs: List[Dict[str, Union[torch.Tensor, List]]],
        weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        outputs_dict = {}
        for k, v in outputs[0].items():
            if isinstance(v, torch.Tensor):
                outputs_dict[k] = torch.einsum(
                    'ip,ip...->p...',
                    weights,
                    torch.stack([o[k] for o in outputs], dim=0),
                )
            elif isinstance(v, list):
                try:
                    outputs_dict[k] = [
                        torch.einsum(
                            'ip,ip...->p...',
                            weights,
                            torch.stack([o[k][i] for o in outputs], dim=0),
                        )
                        for i in range(len(v))
                    ]
                except TypeError:
                    pass
        return outputs_dict

    # =====================================================================
    # Camera output
    # =====================================================================
    @torch.no_grad()
    def get_outputs_for_camera(
        self,
        camera: Union[Cameras, List[Cameras]],
        obb_box: Optional[OrientedBox] = None,
        which: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(camera, list):
            outputs = [
                self.get_outputs_for_camera(c, obb_box=obb_box)
                for c in camera
            ]
            weights = torch.cat(
                [c.metadata['camera_weights'] for c in camera]
            )
            outputs = self._sum_cameras_with_weights(outputs, weights)
            return outputs
        else:
            return self.get_outputs_for_camera_ray_bundle(
                camera.generate_rays(
                    camera_indices=0, keep_shape=True, obb_box=obb_box
                )
            )

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(
        self, camera_ray_bundle: RayBundle
    ) -> Dict[str, torch.Tensor]:
        input_device = camera_ray_bundle.directions.device
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(
                start_idx, end_idx
            )
            ray_bundle = ray_bundle.to(self.device)
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():
                if not isinstance(output, torch.Tensor):
                    continue
                outputs_lists[output_name].append(output.to(input_device))
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(
                image_height, image_width, -1
            )
        return outputs

    def _sum_cameras_with_weights(
        self,
        outputs: List[Dict[str, Union[torch.Tensor, List]]],
        weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        outputs_dict = {}
        for k, v in outputs[0].items():
            if isinstance(v, torch.Tensor):
                outputs_dict[k] = torch.einsum(
                    'i,i...->...',
                    weights,
                    torch.stack([o[k] for o in outputs], dim=0),
                )
            elif isinstance(v, list):
                try:
                    outputs_dict[k] = [
                        torch.einsum(
                            'i,i...->...',
                            weights,
                            torch.stack([o[k][i] for o in outputs], dim=0),
                        )
                        for i in range(len(v))
                    ]
                except TypeError:
                    pass
        return outputs_dict

    # =====================================================================
    # Metrics and losses
    # =====================================================================
    def get_metrics_dict(self, outputs, batch) -> Dict:
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        if self.deformation_field is not None:
            metrics_dict["mean_disp"] = self.deformation_field.mean_disp()
            metrics_dict["max_disp"] = self.deformation_field.max_disp()
            if hasattr(self.deformation_field, 'mean_growth'):
                metrics_dict["mean_growth"] = self.deformation_field.mean_growth()
                metrics_dict["max_growth"] = self.deformation_field.max_growth()

        metrics_dict["flat_field"] = self.flat_field.phi_x.mean().item()
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )
        loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]
        predicted_rgb = predicted_rgb * predicted_rgb.new_ones(1, 3)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        diff_rgb = torch.abs(gt_rgb - predicted_rgb).mean(dim=-1, keepdim=True)
        diff_rgb = colormaps.apply_colormap(diff_rgb)
        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        gt_rgb_m = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb_m = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb_m, predicted_rgb_m)
        ssim = self.ssim(gt_rgb_m, predicted_rgb_m)
        lpips = self.lpips(gt_rgb_m, predicted_rgb_m)

        metrics_dict = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim),
        }
        metrics_dict["lpips"] = float(lpips)

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
            "diff_rgb": diff_rgb,
        }

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
