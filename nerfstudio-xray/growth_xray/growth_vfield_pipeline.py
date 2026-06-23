"""
Growth-aware velocity field pipeline for dendrite 4D reconstruction.
CORRECTED VERSION.

Extends VanillaPipeline with:
1. Growth-specific regularization losses (sparsity, non-negativity, monotonicity, directional)
2. Single-canonical volumetric self-consistency loss at t=T
3. No mismatch_penalty (single canonical, no forward/backward)
4. Growth field logging and visualization

Changes from original VfieldPipeline:
- Removed: mismatch_penalty, volumetric_loss_0, volumetric_loss_1, spatiotemporal mixing
- Added: growth_sparsity, growth_negativity, growth_temporal_monotonicity, growth_directional
- Simplified: volumetric_loss_T (self-consistency at canonical time)
"""
import typing
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, Literal, Optional, Tuple, Type, List

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager, DataManagerConfig, VanillaDataManager,
)
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler
from nerfstudio.utils.rich_utils import CONSOLE
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import defaultdict

from nerf_xray.xray_datamanager import XrayDataManagerConfig
from .growth_vfield_model import GrowthVfieldModel, GrowthVfieldModelConfig
from nerf_xray.objects import Object


@dataclass
class GrowthVfieldPipelineConfig(VanillaPipelineConfig):
    """Configuration for growth-aware pipeline."""

    _target: Type = field(default_factory=lambda: GrowthVfieldPipeline)
    datamanager: DataManagerConfig = field(
        default_factory=lambda: XrayDataManagerConfig
    )
    model: ModelConfig = field(
        default_factory=lambda: GrowthVfieldModelConfig
    )

    # Volumetric supervision (self-consistency at last frame)
    volumetric_supervision: bool = False
    """Enable self-consistency check: at t=T, model prediction should match canonical."""
    volumetric_supervision_start_step: int = 100
    """Start volumetric supervision after this step."""
    volumetric_supervision_coefficient: float = 0.005
    """Weight for volumetric supervision loss."""

    # Growth regularization
    growth_sparsity_coefficient: float = 1e-3
    """L1 sparsity penalty on growth rate (encourages interface-only growth)."""
    growth_sparsity_start_step: int = 0
    """Step to start growth sparsity loss."""

    growth_negativity_coefficient: float = 1e-2
    """Penalty for negative growth (enforces liquid->solid only).
    Set to 0 if remelting should be allowed."""
    growth_negativity_start_step: int = 0
    """Step to start negativity penalty."""

    growth_temporal_monotonicity_coefficient: float = 1e-3
    """Penalty for density decreasing over time at random spatial points."""
    growth_temporal_monotonicity_start_step: int = 500
    """Step to start temporal monotonicity penalty."""
    growth_temporal_monotonicity_every_n_steps: int = 5
    """Compute temporal monotonicity loss every N steps (expensive: 2 ODE integrations).
    FIX: original code computed every step, causing ~40% overhead."""

    growth_directional_coefficient: float = 0.0
    """Bias growth along z-axis (thermal gradient direction).
    0 = no directional bias. Set >0 for directional solidification."""
    growth_directional_start_step: int = 0

    # Velocity field regularization (CRITICAL for single-canonical at high resolution)
    velocity_dc_coefficient: float = 0.0
    """Penalize spatially-uniform (DC) velocity component.
    Prevents global rigid-body drift that occurs with single-canonical 
    architecture under sparse projections (2 angles).
    Recommended: 1e-3 for vel_27+, 0 for vel_6/vel_15 (unnecessary at low res).
    Physics: dendrite growth has no global sample translation."""
    velocity_dc_start_step: int = 0
    """Step to start velocity DC regularization."""

    # ======== MDUB: Monotonic Density Upper Bound ========
    density_upper_bound_coefficient: float = 0.0
    """Enforce ρ(x,t) ≤ ρ_canonical(x). Core anti-drift constraint.
    Physics: solidification irreversibly increases solid fraction.
    Recommended: 5e-3 for vel_6, 1e-3 for vel_27+."""
    density_upper_bound_start_step: int = 200
    """Start after canonical field stabilizes (~200 steps)."""
 
    # ======== BTCC: Backward Temporal Coherence Chain ========
    temporal_coherence_coefficient: float = 0.0
    """Enforce smooth structural evolution between adjacent frames.
    Physics: consecutive frames differ only by small growth + small deformation.
    Recommended: 5e-3 for vel_6, 1e-3 for vel_27+."""
    temporal_coherence_start_step: int = 200
    temporal_coherence_every_n_steps: int = 3
    """Compute every N steps. 3 = good balance of cost vs coverage.
    Each call = 2 ODE integrations + 1 canonical query."""

    # Flat field
    flat_field_loss_multiplier: float = 0.0
    """Multiplier for flat field regularization."""


class GrowthVfieldPipeline(VanillaPipeline):
    """Pipeline for growth-aware dendrite 4D reconstruction.
    
    Training loop:
        1. Sample ray bundle from datamanager (includes time stamps)
        2. Forward through model (velocity warp + growth accumulation)
        3. Compute projection loss (rgb_loss)
        4. Compute growth regularization losses
        5. Optionally compute volumetric self-consistency at t=T
    """

    config: GrowthVfieldPipelineConfig
    model: GrowthVfieldModel

    def __init__(
        self,
        config: GrowthVfieldPipelineConfig,
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

        assert self.datamanager.train_dataset is not None
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
                GrowthVfieldModel,
                DDP(self._model, device_ids=[local_rank], find_unused_parameters=True),
            )
            dist.barrier(device_ids=[local_rank])

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """Compute training losses including growth regularization.
        
        Loss components:
        - rgb_loss: projection reconstruction error (from model)
        - flat_field_loss: flat field regularization
        - growth_sparsity: L1 on growth rate (interface localization)
        - growth_negativity: penalty for negative growth
        - growth_temporal_monotonicity: density should not decrease over time
        - volumetric_loss_T: self-consistency at canonical time
        """
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        # Flat field loss
        loss_dict['flat_field_loss'] = self._get_flat_field_penalty()

        # Growth regularization losses
        if (hasattr(self.model.deformation_field, 'growth_nn') 
            and self.model.deformation_field.growth_nn is not None) \
            or getattr(self.model.deformation_field, '_use_shared_encoder', False):
            self._add_growth_losses(loss_dict, step)

        # Velocity DC regularization (prevents horizontal shift at high resolution)
        if (
            self.config.velocity_dc_coefficient > 0
            and step > self.config.velocity_dc_start_step
        ):
            loss_dict['velocity_dc_loss'] = (
                self.config.velocity_dc_coefficient
                * self._get_velocity_dc_penalty()
            )

        # ── MDUB: Monotonic Density Upper Bound ──
        if (
            self.config.density_upper_bound_coefficient > 0
            and step > self.config.density_upper_bound_start_step
        ):
            loss_dict['density_upper_bound'] = (
                self.config.density_upper_bound_coefficient
                * self._compute_density_upper_bound_loss(npts=2048)
            )
 
        # ── BTCC: Backward Temporal Coherence Chain ──
        if (
            self.config.temporal_coherence_coefficient > 0
            and step > self.config.temporal_coherence_start_step
            and step % self.config.temporal_coherence_every_n_steps == 0
        ):
            loss_dict['temporal_coherence'] = (
                self.config.temporal_coherence_coefficient
                * self._compute_temporal_coherence_loss(npts=2048)
            )

        # Volumetric self-consistency at last frame
        if (
            self.config.volumetric_supervision
            and step > self.config.volumetric_supervision_start_step
        ):
            vol_loss = self._calculate_volumetric_loss(time=1.0)
            loss_dict['volumetric_loss_T'] = (
                self.config.volumetric_supervision_coefficient * vol_loss
            )

        return model_outputs, loss_dict, metrics_dict

    def _add_growth_losses(self, loss_dict: Dict, step: int):
        """Compute and add growth-specific regularization losses.
        
        All losses except temporal_monotonicity share one growth_rate evaluation
        at random (x, t) for efficiency.
        """
        npts = 2048
        device = self.device
        
        # Sample random points in scene box
        pos = (torch.rand(npts, 3, device=device) - 0.5) * 1.4
        t_rand = torch.rand(1, device=device)
        x0, x1, x2 = pos[:, 0], pos[:, 1], pos[:, 2]

        # Compute growth rate at random points (one forward pass)
        g = self.model.deformation_field.growth_rate(x0, x1, x2, t_rand)

        # 1. Growth sparsity (L1): growth should be localized at interface
        if (
            step > self.config.growth_sparsity_start_step
            and self.config.growth_sparsity_coefficient > 0
        ):
            loss_dict['growth_sparsity'] = (
                self.config.growth_sparsity_coefficient * g.abs().mean()
            )

        # 2. Growth non-negativity: liquid->solid is irreversible
        if (
            step > self.config.growth_negativity_start_step
            and self.config.growth_negativity_coefficient > 0
        ):
            loss_dict['growth_negativity'] = (
                self.config.growth_negativity_coefficient
                * torch.relu(-g).mean()
            )

        # 3. Directional growth bias (z-axis aligned with thermal gradient)
        if (
            step > self.config.growth_directional_start_step
            and self.config.growth_directional_coefficient > 0
        ):
            # Penalize large growth at z < 0 (already solidified region)
            # Encourage growth at z > 0 (solidification front)
            z_weight = torch.sigmoid(-5.0 * x2)  # high at negative z
            loss_dict['growth_directional'] = (
                self.config.growth_directional_coefficient
                * (g * z_weight).abs().mean()
            )

        # 4. Temporal monotonicity: density should not decrease over time
        # FIX: only compute every N steps (expensive: 2 full ODE integrations)
        if (
            step > self.config.growth_temporal_monotonicity_start_step
            and self.config.growth_temporal_monotonicity_coefficient > 0
            and step % self.config.growth_temporal_monotonicity_every_n_steps == 0
        ):
            mono_loss = self._compute_temporal_monotonicity_loss(npts=512)
            loss_dict['growth_temporal_monotonicity'] = (
                self.config.growth_temporal_monotonicity_coefficient * mono_loss
            )

    def _get_velocity_dc_penalty(self) -> Tensor:
        """Penalize the spatially-uniform (DC) component of the velocity field.
        
        At each sampled time, the spatial mean of v(x, t) should be near zero.
        A nonzero mean implies global rigid-body translation, which is:
          - Physically wrong (sample holder is stationary)
          - Unobservable from 2 projection angles (underconstrained)
          - Causes systematic spatial shift in reconstruction
        
        This is especially critical for single-canonical architectures where
        there is no forward/backward mutual anchoring to prevent drift.
        
        Cost: ~5 velocity evaluations, negligible vs ODE integration.
        """
        device = self.device
        n_points = 512
        
        # Random spatial positions in scene box
        pos = (torch.rand(n_points, 3, device=device) - 0.5) * 1.4
        x0, x1, x2 = pos[:, 0], pos[:, 1], pos[:, 2]
        
        # Sample a few random times
        n_times = 5
        times = torch.rand(n_times, device=device)
        
        penalty = torch.zeros(1, device=device)
        for t in times:
            u = self.model.deformation_field.velocity(x0, x1, x2, t)  # [N, 3]
            mean_u = u.mean(dim=0)  # [3] spatial mean
            penalty = penalty + mean_u.pow(2).sum()
        
        return penalty / n_times

    def _compute_density_upper_bound_loss(self, npts: int = 2048) -> Tensor:
        """Monotonic Density Upper Bound (MDUB).

        Constraint: ρ(x, t) ≤ ρ_canonical(x)  for random (x, t).

        How it prevents drift — code-level causality:
            1. Velocity field creates displacement Δx at time t
            2. Canonical field is queried at shifted position: ρ_can(x + Δx)
            3. If Δx maps x to a high-density region of canonical,
               model density ρ(x,t) becomes artificially high
            4. But MDUB checks canonical density at ORIGINAL position x
            5. If ρ_canonical(x) is low (e.g., liquid region) but ρ(x,t) is high
               → violation → gradient penalizes Δx back through weight_nn

        Gradient targets:
            ∂L/∂θ_velocity  — reduces displacement that causes violations
            ∂L/∂θ_growth    — adjusts growth_factor to respect bound
            ρ_canonical is detached — canonical field stays stable
        """
        device = self.device
        pos = (torch.rand(npts, 3, device=device) - 0.5) * 1.4
        t = (torch.rand(1, device=device) * 0.92).item()

        model_density = self.model.get_density_from_pos(pos, time=t)

        canonical_density = self.model.field.get_density_from_pos(
            pos, deformation_field=None, time=1.0
        )
        if canonical_density is None:
            return torch.zeros(1, device=device)
        canonical_density = canonical_density.squeeze().detach()

        violation = torch.relu(model_density - canonical_density)
        return violation.mean()

    def _compute_temporal_coherence_loss(self, npts: int = 2048) -> Tensor:
        """Backward Temporal Coherence Chain (BTCC).

        Enforces smooth structural evolution between adjacent time frames
        through a stochastic chain anchored at canonical time T.

        Per training step, ONE random link (t, t+δ) of the chain is sampled.
        Over 1000 calls, all ~29 links are sampled ~34 times each — sufficient
        for stochastic coverage of the full T→0 chain.

        Two components:
          A. Structural NCC — density PATTERN similarity in structure regions
             prevents "dendrite at T, blob at T-1"
          B. Bounded density change — density VALUE change ≤ ε(t) in structure
             regions, where ε(t) is tighter near T (less change expected)

        The canonical density field (detached) provides structure weighting:
        constraints are focused on where canonical has structure, not on empty
        space where density is trivially zero at all times.

        Why this works with the growth architecture:
            The growth field already encodes "how much to remove from T".
            BTCC constrains the SMOOTHNESS of this removal across frames:
            - Growth rate per frame must be bounded (component B)
            - The spatial pattern of removal must be gradual (component A)
            Without BTCC, growth+velocity can produce discontinuous jumps.

        Gradient targets:
            ∂L/∂θ_velocity  — smooths velocity field over time
            ∂L/∂θ_growth    — bounds growth rate per frame interval
            rho_t1, rho_t2 NOT detached — both frames pulled to consistency
            rho_canonical IS detached — canonical stays stable
        """
        device = self.device

        # ── Sample spatial points ──
        pos = (torch.rand(npts, 3, device=device) - 0.5) * 1.4

        # ── Sample ONE random chain link (t, t+δ) ──
        # δ = frame interval, aligned with actual data frame rate
        num_frames = getattr(self.config, 'num_frames', 30)
        delta = 1.0 / max(num_frames - 1, 1)
        t1_val = (torch.rand(1, device=device) * (1.0 - delta - 0.02)).item()
        t2_val = t1_val + delta

        # ── Query densities ──
        # Both through full implicit pipeline (ODE + growth + canonical)
        # Neither is detached: gradients flow to velocity + growth
        rho_t1 = self.model.get_density_from_pos(pos, time=t1_val)
        rho_t2 = self.model.get_density_from_pos(pos, time=t2_val)

        # ── Canonical density for structure weighting (detached) ──
        rho_can = self.model.field.get_density_from_pos(
            pos, deformation_field=None, time=1.0
        )
        if rho_can is None:
            return torch.zeros(1, device=device)
        rho_can = rho_can.squeeze().detach()

        can_max = rho_can.max()
        if can_max < 1e-8:
            return torch.zeros(1, device=device)

        # Soft structure mask: smooth transition at 2% of max canonical density
        # Fully differentiable w.r.t. rho_t1, rho_t2 (canonical is detached)
        w = torch.sigmoid(20.0 * (rho_can / can_max - 0.02))

        # ── Component A: Structural NCC in canonical-support ──
        # Purpose: prevent spatial pattern discontinuity between frames
        sig_mask = w > 0.5
        n_sig = sig_mask.sum().item()

        if n_sig > 50:
            r1 = rho_t1[sig_mask]
            r2 = rho_t2[sig_mask]
            r1c = r1 - r1.mean()
            r2c = r2 - r2.mean()
            ncc = (r1c * r2c).sum() / (
                torch.sqrt((r1c ** 2).sum() + 1e-8)
                * torch.sqrt((r2c ** 2).sum() + 1e-8)
            )
            ncc_loss = torch.clamp(1.0 - ncc, min=0.0)
        else:
            ncc_loss = torch.zeros(1, device=device)

        # ── Component B: Canonical-weighted bounded density change ──
        # Purpose: density change per frame must be small in structure regions
        #
        # Time-dependent tolerance: ε(t) = ε₀ × (1 - 0.7t)
        #   t=0 (early solidification): ε₀        — allow larger changes
        #   t≈1 (near terminal):        0.3 × ε₀  — very tight
        eps_base = 0.12  # max density change per frame at t=0
        eps = eps_base * (1.0 - 0.7 * t1_val)

        density_change = (rho_t2 - rho_t1).abs()
        excess = torch.relu(density_change - eps)
        change_loss = (excess * w).mean()

        # ── Combined: 50/50 split ──
        return 0.5 * ncc_loss + 0.5 * change_loss
    
    def _compute_temporal_monotonicity_loss(self, npts: int = 512) -> Tensor:
        """Penalize cases where density decreases from t to t+dt.
        
        Sample random (x, t) pairs, query density at t and t+dt,
        penalize max(0, rho(t) - rho(t+dt)).
        
        Note: This requires 2 full ODE integrations (expensive).
        Use growth_temporal_monotonicity_every_n_steps to control frequency.
        """
        device = self.device
        pos = (torch.rand(npts, 3, device=device) - 0.5) * 1.4
        t1 = torch.rand(1, device=device) * 0.9  # t in [0, 0.9]
        t2 = t1 + 0.1  # t + dt

        density_t1 = self.model.get_density_from_pos(pos, time=t1.item())
        density_t2 = self.model.get_density_from_pos(pos, time=t2.item())

        # Penalty for density decrease
        violation = torch.relu(density_t1 - density_t2)
        return violation.mean()

    def _calculate_volumetric_loss(self, time: float = 1.0) -> Tensor:
        """Compute volumetric self-consistency loss at canonical time.
        
        At t=T, the ODE integration is identity (t→T with t=T), so:
            warped_pos = original_pos (no warp)
            growth_accum = 0 (no integration)
            model_density = canonical_density
        
        This loss enforces that the deformation field + growth field
        jointly produce identity at the reference time, preventing drift.
        
        NOTE: This is NOT GT supervision against an external volume.
        If you have external GT at other timesteps, implement a separate function.
        """
        npts = self.config.datamanager.train_num_rays_per_batch * 32
        pos = (torch.rand(npts, 3, device=self.device) - 0.5) * 1.4

        # Model prediction at time T (goes through full pipeline)
        pred_density = self.model.get_density_from_pos(pos, time=time)

        # GT density: raw canonical field without deformation
        gt_density = self.model.field.get_density_from_pos(
            pos, deformation_field=None, time=time
        )
        if gt_density is not None:
            gt_density = gt_density.squeeze()
        else:
            return torch.zeros(1, device=self.device)

        # Normalized cross-correlation loss
        ncc = self._normed_correlation(pred_density, gt_density)
        return 1.0 - ncc

    @staticmethod
    def _normed_correlation(x: Tensor, y: Tensor) -> Tensor:
        """Normalized cross-correlation between two vectors."""
        mux = x.mean()
        muy = y.mean()
        dx = x - mux
        dy = y - muy
        return torch.sum(dx * dy) / (
            torch.sqrt(dx.pow(2).sum() * dy.pow(2).sum()) + 1e-8
        )

    def _get_flat_field_penalty(self) -> Tensor:
        return (
            -self.config.flat_field_loss_multiplier
            * self.model.flat_field.phi_x.mean()
        )

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        metrics_dict.update(
            {'flat_field': self.model.flat_field.phi_x.mean()}
        )
        # Log growth stats
        if hasattr(self.model.deformation_field, 'mean_growth'):
            metrics_dict['mean_growth'] = self.model.deformation_field.mean_growth()
            metrics_dict['max_growth'] = self.model.deformation_field.max_growth()

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_average_eval_image_metrics(
        self,
        step: Optional[int] = None,
        output_path: Optional[Path] = None,
        get_std: bool = False,
        **kwargs,
    ):
        self.eval()
        metrics_dict_list = []
        assert isinstance(
            self.datamanager,
            (VanillaDataManager, ParallelDataManager, FullImageDatamanager),
        )
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(
                "[green]Evaluating all eval images...", total=num_images
            )
            for camera, batch in self.datamanager.fixed_indices_eval_dataloader:
                inner_start = time()
                outputs = self.model.get_outputs_for_camera(camera=camera)
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, _ = self.model.get_image_metrics_and_images(
                    outputs, batch
                )
                loss_dict = self.model.get_loss_dict(outputs, batch, metrics_dict)
                for key in loss_dict.keys():
                    if isinstance(loss_dict[key], torch.Tensor):
                        loss_dict[key] = loss_dict[key].item()

                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = (
                    num_rays / (time() - inner_start)
                ).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (
                    metrics_dict["num_rays_per_sec"] / (height * width)
                ).item()

                image_idx = batch['image_idx']
                img_filename = self.datamanager.eval_dataset.image_filenames[
                    image_idx
                ]
                metrics_dict["image_name"] = img_filename.as_posix()
                metrics_dict["image_time"] = camera.times.item()
                metrics_dict.update(loss_dict)
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)

        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if isinstance(metrics_dict_list[0][key], str):
                continue
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor(
                        [md[key] for md in metrics_dict_list]
                    )
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(
                        torch.tensor(
                            [md[key] for md in metrics_dict_list]
                        )
                    )
                )
        metrics_dict['metrics_list'] = metrics_dict_list
        self.train()
        return metrics_dict
    #6.4
    def eval_along_plane(
        self,
        target: Literal['field', 'datamanager', 'both'] = 'field',
        plane='xy',
        distance=0.0,
        fn=None,
        engine='cv',
        resolution=500,
        rhomax=1.0,
        time=0.0,
        which=None,  # ignored for growth model, kept for API compatibility
    ):
        """Evaluate density along a 2D plane slice.
        
        Used by exporter.py for volume export.
        """
        import cv2 as cv
        
        a = torch.linspace(-1, 1, resolution, device=self.device)
        b = torch.linspace(-1, 1, resolution, device=self.device)
        A, B = torch.meshgrid(a, b, indexing='ij')
        C = distance * torch.ones_like(A)
        if plane == 'xy':
            pos = torch.stack([A, B, C], dim=-1)
        elif plane == 'yz':
            pos = torch.stack([C, A, B], dim=-1)
        elif plane == 'xz':
            pos = torch.stack([A, C, B], dim=-1)

        if target in ['field', 'both']:
            with torch.no_grad():
                pred_density = self._model.get_density_from_pos(
                    pos, time=time
                ).squeeze()
            pred_density = pred_density.cpu().numpy() / rhomax

        if target in ['datamanager', 'both']:
            pos_shape = pos.shape
            obj_density = self.datamanager.object.density(
                pos.view(-1, 3)
            ).view(pos_shape[:-1])
            max_density = self.datamanager.object.max_density
            obj_density = obj_density.cpu().numpy() / max_density

        if target == 'both':
            density = np.concatenate([obj_density, pred_density], axis=1)
        elif target == 'field':
            density = pred_density
        elif target == 'datamanager':
            density = obj_density

        if engine == 'matplotlib':
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 6) if target != 'both' else (12, 6))
            plt.imshow(density, extent=[-1, 1, -1, 1], origin='lower',
                      cmap='gray', vmin=0, vmax=1)
            if fn is not None:
                plt.savefig(fn)
            plt.close()
        elif engine in ['cv', 'opencv']:
            density = np.clip(density, 0, 1)
            density = (density * 255).astype(np.uint8)
            if fn is not None:
                if isinstance(fn, Path):
                    fn = fn.as_posix()
                cv.imwrite(fn, density)
        elif engine == 'numpy':
            return density
        else:
            raise ValueError(f"Invalid engine {engine}")
