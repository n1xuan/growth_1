"""
Model for the velocity field method.
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
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    MSELoss, distortion_loss, interlevel_loss, orientation_loss,
    pred_normal_loss, scale_gradients_by_distance_squared)
from nerfstudio.model_components.ray_samplers import (ProposalNetworkSampler,
                                                      UniformSampler)
from nerfstudio.model_components.renderers import (AccumulationRenderer,
                                                   DepthRenderer,
                                                   NormalsRenderer,
                                                   RGBRenderer)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
from nerfstudio.models.nerfacto import (  # for subclassing Nerfacto model
    NerfactoModel, NerfactoModelConfig)
from nerfstudio.utils import colormaps
from nerfstudio.data.scene_box import OrientedBox, SceneBox
from torch import Tensor
from torch.nn import Parameter

from nerf_xray.field_mixers import FieldMixerConfig, SpatiotemporalMixingRenderer, FieldMixer
from .deformation_fields import (AffineTemporalDeformationField,
                                 BsplineTemporalDeformationField1d,
                                 BsplineTemporalDeformationField3d,
                                 BsplineTemporalDeformationField3dConfig,
                                 BsplineDeformationField3d,
                                 BSplineField1d,
                                 DeformationFieldConfig,
                                 IdentityDeformationField,
                                 MLPDeformationField)
from .canonical_field import CanonicalNerfField, PlaceHolderField
from .xray_renderer import AttenuationRenderer
    
@dataclass
class VfieldModelConfig(NerfactoModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: VfieldModel)
    train_density_field: bool = True
    """whether to train the density field"""
    train_deformation_field: bool = False
    """whether to train the deformation field"""
    deformation_field: DeformationFieldConfig = field(default_factory=lambda: DeformationFieldConfig)
    """Forward deformation field"""
    field_weighing: FieldMixerConfig = field(default_factory=lambda: FieldMixerConfig)
    """Field weighing"""
    flat_field_value: float = 0.0
    """initial value of flat field"""
    flat_field_trainable: bool = False
    """trainable background color"""
    train_field_weighing: bool = True
    """whether to train field weighing"""
    direction: Literal['forward','backward','both'] = 'both'
    """direction in which to fit the model"""
    camera_optimizer = None
    """Config of the camera optimizer to use"""
    disable_mixing: bool = False
    """If True, the forward and backward canonical models will alternate as opposed to combine"""

class VfieldModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: VfieldModelConfig
    field_weighing: FieldMixer

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        appearance_embedding_dim = self.config.appearance_embed_dim if self.config.use_appearance_embedding else 0

        # Fields
        if self.config.direction in ['forward', 'both']:
            self.field_f = CanonicalNerfField(
                self.scene_box.aabb,
                hidden_dim=self.config.hidden_dim,
                num_levels=self.config.num_levels,
                max_res=self.config.max_res,
                base_res=self.config.base_res,
                features_per_level=self.config.features_per_level,
                log2_hashmap_size=self.config.log2_hashmap_size,
                hidden_dim_color=self.config.hidden_dim_color,
                hidden_dim_transient=self.config.hidden_dim_transient,
                spatial_distortion=scene_contraction,
                num_images=self.num_train_data,
                use_pred_normals=self.config.predict_normals,
                use_average_appearance_embedding=self.config.use_average_appearance_embedding,
                appearance_embedding_dim=appearance_embedding_dim,
                average_init_density=self.config.average_init_density,
                implementation=self.config.implementation,
            )
        else:
            self.field_f = PlaceHolderField()

        if self.config.direction in ['backward', 'both']:
            self.field_b = CanonicalNerfField(
                self.scene_box.aabb,
                hidden_dim=self.config.hidden_dim,
                num_levels=self.config.num_levels,
                max_res=self.config.max_res,
                base_res=self.config.base_res,
                features_per_level=self.config.features_per_level,
                log2_hashmap_size=self.config.log2_hashmap_size,
                hidden_dim_color=self.config.hidden_dim_color,
                hidden_dim_transient=self.config.hidden_dim_transient,
                spatial_distortion=scene_contraction,
                num_images=self.num_train_data,
                use_pred_normals=self.config.predict_normals,
                use_average_appearance_embedding=self.config.use_average_appearance_embedding,
                appearance_embedding_dim=appearance_embedding_dim,
                average_init_density=self.config.average_init_density,
                implementation=self.config.implementation,
            )
        else:
            self.field_b = PlaceHolderField()

        self.deformation_field = self.config.deformation_field.setup()
        self.field_weighing = self.config.field_weighing.setup()

        # train density or deformation field or field weighing
        if not self.config.train_density_field:
            self.field_f.requires_grad_(False)
            self.field_b.requires_grad_(False)
        if not self.config.train_deformation_field:
            self.deformation_field.requires_grad_(False)
        if not self.config.train_field_weighing:
            self.field_weighing.requires_grad_(False)

        self.camera_optimizer = None
        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                average_init_density=self.config.average_init_density,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    average_init_density=self.config.average_init_density,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        # self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="median")
        self.renderer_expected_depth = DepthRenderer(method="expected")
        self.renderer_normals = NormalsRenderer()
        self.renderer_attenuation = AttenuationRenderer(
            background_color=self.config.background_color,
            )
        self.renderer_rgb = self.renderer_attenuation
        self.spatiotemporal_mixing_renderer = SpatiotemporalMixingRenderer()
        
        ff = self.kwargs['metadata'].get('flat_field', None)
        if ff is not None:
            _ff = -np.log(ff)
            CONSOLE.print(f"Using flat field from metadata: {ff:.3f} -> {_ff:.3f}")
            ff = _ff
        else:
            ff = self.config.flat_field_value
        # self.flat_field = torch.nn.Parameter(
        #     torch.tensor(ff, dtype=torch.float32), 
        #     self.config.flat_field_trainable
        # )
        self.flat_field = BSplineField1d(
            torch.nn.parameter.Parameter(ff*torch.ones(10)), 
            support_outside=True, 
            support_range=(0,1)
        )
        if not self.config.flat_field_trainable:
            self.flat_field.requires_grad_(False)

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()
        self.step = 0
        # metrics
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import \
            LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field_f.parameters())
        param_groups["fields"].extend(list(self.field_b.parameters()))
        param_groups['fields'].extend(list(self.deformation_field.parameters()))
        param_groups['field_weighing'] = list(self.field_weighing.parameters())
        # trainable background color
        # if self.config.flat_field_trainable:
        param_groups["flat_field"] = list(self.flat_field.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                self.step = step
                train_frac = np.clip(step / N, 0, 1)
                self.step = step

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
    
    def forward(self, ray_bundle: Union[List[RayBundle], RayBundle, Cameras, Tensor], which: Optional[Literal['forward','backward','mixed']] = None) -> Dict[str, Union[torch.Tensor, List]]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included or positions for volumetric training
        """
        if isinstance(ray_bundle, list):
            outputs = [self.forward(rb, which=which) for rb in ray_bundle] # list of dicts
            # weighted average
            weights = torch.cat([rb.metadata['camera_weights'] for rb in ray_bundle], dim=1)
            # do not normalize. Assume they come in normalized as they should be
            # weights = weights / weights.sum(dim=1, keepdim=True) # [num_rays, num_cameras] # do not normalize
            weights = weights.permute(1,0) # [num_cameras, num_rays]
            outputs = self.sum_outputs_with_weights(outputs, weights)
            return outputs
        else:
            if self.collider is not None:
                ray_bundle = self.collider(ray_bundle)
            return self.get_outputs(ray_bundle, which=which)
    
    def sum_outputs_with_weights(
        self,
        outputs: List[Dict[str, Union[torch.Tensor, List]]],
        weights: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        outputs_dict = {}
        for k, v in outputs[0].items():
            if isinstance(v, torch.Tensor):
                outputs_dict[k] = torch.einsum('ip,ip...->p...', weights, torch.stack([o[k] for o in outputs], dim=0))
            elif isinstance(v, list):
                try:
                    outputs_dict[k] = [torch.einsum('ip,ip...->p...', weights, torch.stack([o[k][i] for o in outputs], dim=0)) for i in range(len(v))]
                except TypeError: # does not work for RaySamples. Could take middle one but probably best to skip downstream applications
                    pass
        return outputs_dict
    
    def sum_cameras_with_weights(
        self,
        outputs: List[Dict[str, Union[torch.Tensor, List]]],
        weights: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        outputs_dict = {}
        for k, v in outputs[0].items():
            if isinstance(v, torch.Tensor):
                outputs_dict[k] = torch.einsum('i,i...->...', weights, torch.stack([o[k] for o in outputs], dim=0))
            elif isinstance(v, list):
                try:
                    outputs_dict[k] = [torch.einsum('i,i...->...', weights, torch.stack([o[k][i] for o in outputs], dim=0)) for i in range(len(v))]
                except TypeError: # does not work for RaySamples. Could take middle one but probably best to skip downstream applications
                    pass
        return outputs_dict

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Union[Cameras, List[Cameras]], obb_box: Optional[OrientedBox] = None, which: Optional[Literal['forward','backward','mixed']] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Assumes a ray-based model.

        Args:
            camera: generates raybundle
        """
        if isinstance(camera, list):
            outputs = [self.get_outputs_for_camera(c, obb_box=obb_box, which=which) for c in camera]
            weights = torch.cat([c.metadata['camera_weights'] for c in camera])
            outputs = self.sum_cameras_with_weights(outputs, weights)
            return outputs
        else:
            return self.get_outputs_for_camera_ray_bundle(
                camera.generate_rays(camera_indices=0, keep_shape=True, obb_box=obb_box),
                which=which
            )

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle, which: Optional[Literal['forward','backward','mixed']] = None) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        input_device = camera_ray_bundle.directions.device
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            # move the chunk inputs to the model device
            ray_bundle = ray_bundle.to(self.device)
            outputs = self.forward(ray_bundle=ray_bundle, which=which)
            for output_name, output in outputs.items():  # type: ignore
                if not isinstance(output, torch.Tensor):
                    # TODO: handle lists of tensors as well
                    continue
                # move the chunk outputs from the model device back to the device of the inputs.
                outputs_lists[output_name].append(output.to(input_device))
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs

    def mix_two_fields(self, field_0_outputs: Union[Dict, Tensor], field_1_outputs: Union[Dict, Tensor], alphas: Union[Tensor, float]) -> Union[Dict, Tensor]:
        if isinstance(field_0_outputs, Tensor):
            assert isinstance(field_1_outputs, Tensor)
            field_outputs = (1-alphas) * field_0_outputs + alphas * field_1_outputs
        elif isinstance(field_0_outputs, dict):
            field_outputs = {}
            for key in field_0_outputs:
                field_outputs[key] = (1-alphas) * field_0_outputs[key] + alphas * field_1_outputs[key]
        return field_outputs
        
    def get_fields_mismatch_penalty(self, reduction: Literal['mean','sum','none'] = 'sum', npoints: Optional[int] = None):
        # sample time
        times = torch.linspace(0, 1, 11, device=self.device) # 0 to 1
        if self.training: # perturb
            times += (torch.rand(11, device=self.device)-0.5)*0.1
        alphas = self.get_mixing_coefficient(times)
        # cost = (alphas * (1-alphas)).detach() # should we detach or no?
        cost = torch.sigmoid(50*(alphas*(1-alphas)-0.2))#.detach()
        diffs = []
        if npoints is None:
            npoints = 1<<13 # 8096
        for i,t in enumerate(times):
            pos = (2*torch.rand((npoints, 3), device=self.device) - 1.0) * 0.7 # +0.7 to -0.7
            diff = self.get_density_difference(pos, t.item()).pow(2).mean().view(1)
            if self.training:
                diff = diff * cost[i]
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

    @contextmanager
    def empty_context_manager(self):
        yield

    def get_density_from_pos(
        self, positions: Tensor, time: Optional[Union[Tensor, float]] = None, which: Optional[Literal['forward','backward','mixed']] = None
    ) -> Tensor:
        if time is None:
            time = positions.new_zeros(1)
        elif isinstance(time, float):
            time = positions.new_ones(1)*time
        else:
            raise ValueError(f'`time` of type {type(time)}')
        if self.training:
            cm = self.empty_context_manager
        else:
            cm = torch.no_grad

        with cm():
            if which!='backward':
                density_0 = self.field_f.get_density_from_pos(positions, deformation_field=lambda x,t: self.deformation_field(x,t,0.0), time=time)
                if density_0 is not None:
                    density_0 = density_0.squeeze()
            if which!='forward':
                density_1 = self.field_b.get_density_from_pos(positions, deformation_field=lambda x,t: self.deformation_field(x,t,1.0), time=time)
                if density_1 is not None:
                    density_1 = density_1.squeeze()
        if which=='forward':
            return density_0
        if which=='backward':
            return density_1
        assert density_0 is not None and density_1 is not None
        alphas = self.field_weighing.get_mixing_coefficient(positions, time, self.step).squeeze()
        density = self.mix_two_fields(density_0, density_1, alphas) # type: ignore
        return density


    def get_density_difference(
        self, positions: Tensor, time: Optional[Union[Tensor, float]] = None
    ) -> Tensor:
        if self.config.direction != 'both':
            return positions.new_zeros(positions.shape[:-1])

        if time is None:
            time = positions.new_zeros(1)
        elif isinstance(time, float):
            time = positions.new_ones(1)*time
        else:
            raise ValueError(f'`time` of type {type(time)}')
        density_f = self.field_f.get_density_from_pos(positions, deformation_field=lambda x,t: self.deformation_field(x,t,0.0), time=time).squeeze()
        density_b = self.field_b.get_density_from_pos(positions, deformation_field=lambda x,t: self.deformation_field(x,t,1.0), time=time).squeeze()
        return density_f - density_b

    def get_outputs(self, ray_bundle: RayBundle, which: Optional[Literal['forward','backward','mixed']] = None):
        # apply the camera optimizer pose tweaks
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        alphas = acc_alpha= None
        if self.config.disable_mixing and self.training:
            if self.step%10<5:
                field_outputs = self.field_f.forward(ray_samples, compute_normals=self.config.predict_normals, deformation_field=lambda x,t: self.deformation_field(x,t,0.0))
            else:
                field_outputs = self.field_b.forward(ray_samples, compute_normals=self.config.predict_normals, deformation_field=lambda x,t: self.deformation_field(x,t,1.0))
        else:
            if which=='forward':
                field_outputs = self.field_f.forward(ray_samples, compute_normals=self.config.predict_normals, deformation_field=lambda x,t: self.deformation_field(x,t,0.0))
            elif which=='backward':
                field_outputs = self.field_b.forward(ray_samples, compute_normals=self.config.predict_normals, deformation_field=lambda x,t: self.deformation_field(x,t,1.0))
            else:
                field_f_outputs = self.field_f.forward(ray_samples, compute_normals=self.config.predict_normals, deformation_field=lambda x,t: self.deformation_field(x,t,0.0))
                field_b_outputs = self.field_b.forward(ray_samples, compute_normals=self.config.predict_normals, deformation_field=lambda x,t: self.deformation_field(x,t,1.0))
                alphas = self.field_weighing.get_mixing_coefficient(ray_samples.frustums.get_positions(), ray_samples.times, self.step)
                field_outputs = self.mix_two_fields(field_f_outputs, field_b_outputs, alphas)
        
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)
        attenuation = self.renderer_attenuation(densities=field_outputs[FieldHeadNames.DENSITY], ray_samples=ray_samples)
        flat_field = self.flat_field(ray_bundle.times.view(-1)).view(-1,1)
        rgb = self.renderer_attenuation.merge_flat_field(attenuation, flat_field) * attenuation.new_ones(1,3)
        if alphas is not None and not self.training and not self.config.disable_mixing:
            acc_alpha = self.spatiotemporal_mixing_renderer(alphas, ray_samples, field_outputs[FieldHeadNames.DENSITY])

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
            "attenuation": attenuation,
        }
        if acc_alpha is not None:
            outputs["acc_alpha"] = acc_alpha

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs

    def get_metrics_dict(self, outputs, batch) -> Dict:
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        if self.deformation_field is not None:
            metrics_dict["mean_disp"] = self.deformation_field.mean_disp()
            metrics_dict['max_disp'] = self.deformation_field.max_disp()
        # metrics_dict['mismatch_penalty'] = self.get_fields_mismatch_penalty()
        metrics_dict['flat_field'] = self.flat_field.phi_x.mean().item()
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
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)
        predicted_rgb = predicted_rgb * predicted_rgb.new_ones(1,3)
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

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {
            "img": combined_rgb, 
            "accumulation": combined_acc, 
            "depth": combined_depth, 
            "diff_rgb": diff_rgb,
        }
        if "acc_alpha" in outputs:
            images_dict["acc_alpha"] = colormaps.apply_colormap(outputs["acc_alpha"], colormaps.ColormapOptions(normalize=False))

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict