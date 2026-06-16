"""
Multi-resolution refinement for growth-aware velocity field.
v5.1: Full shared-encoder support with proper distillation.

FIXES vs v5.0:
- backbone feature loss now handles width mismatch (old_width != new_width)
  via a learnable linear projection layer
- temporal coherence loss also uses projection when needed

Distillation losses:
  Loss 1: velocity(x,t) output matching — field-level, always active
  Loss 2: growth_rate(x,t) output matching — field-level, when growth enabled
  Loss 3: backbone feature matching — intermediate representation alignment
           Uses projection when backbone widths differ
  Loss 4: temporal coherence — dense time grid feature continuity (phase 2)
  Loss 5: e2e ODE trajectory matching (phase 2)
"""
from typing import Optional, Literal
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np
from nerfstudio.engine.trainer import TrainerConfig
import copy
import matplotlib.pyplot as plt
from tqdm import trange

from growth_deformation_fields import (
    GrowthAwareVelocityField3d,
    GrowthAwareVelocityField3dConfig,
)


def load_growth_field(
    ckpt_path: Path,
    old_config: GrowthAwareVelocityField3dConfig,
):
    """Load growth-aware deformation field from checkpoint.

    Returns:
        field: loaded GrowthAwareVelocityField3d
        key_map: dict mapping short_key -> full pipeline key
        pipeline_prefix: auto-discovered prefix string
    """
    print(f'Loading from {ckpt_path}')
    data = torch.load(ckpt_path, weights_only=False)
    _data = {}
    key_map = {}
    pipeline_prefix = None

    for key in data['pipeline'].keys():
        if 'deformation_field.' in key:
            idx = key.index('deformation_field.')
            if pipeline_prefix is None:
                pipeline_prefix = key[:idx + len('deformation_field.')]
            short_key = key[idx + len('deformation_field.'):]
            _data[short_key] = data['pipeline'][key]
            key_map[short_key] = key

    if pipeline_prefix is None:
        raise ValueError(
            'No deformation_field keys found in checkpoint. '
            f'Available keys sample: {list(data["pipeline"].keys())[:10]}'
        )

    field = old_config.setup()
    missing, unexpected = field.load_state_dict(_data, strict=False)
    if missing:
        print(f'  Warning: missing keys: {missing}')
    if unexpected:
        print(f'  Warning: unexpected keys: {unexpected}')

    return field, key_map, pipeline_prefix


def _has_growth(field: GrowthAwareVelocityField3d) -> bool:
    return (
        getattr(field, '_use_shared_encoder', False)
        or (hasattr(field, 'growth_nn') and field.growth_nn is not None)
    )


def _has_shared_encoder(field: GrowthAwareVelocityField3d) -> bool:
    return getattr(field, '_use_shared_encoder', False)


def _get_backbone_width(field: GrowthAwareVelocityField3d) -> Optional[int]:
    """Get the output width of the shared backbone, or None if not shared."""
    if not _has_shared_encoder(field):
        return None
    # shared_backbone is Sequential: Linear→SELU→Linear→SELU
    # The last Linear's out_features is the backbone width
    for module in reversed(list(field.shared_backbone.modules())):
        if isinstance(module, nn.Linear):
            return module.out_features
    return None


class FeatureProjector(nn.Module):
    """Learnable projection from old backbone width to new backbone width.
    
    Used when old_width != new_width to enable backbone feature matching.
    Initialized as a least-squares pseudo-projection (not identity).
    """
    def __init__(self, old_width: int, new_width: int):
        super().__init__()
        self.proj = nn.Linear(old_width, new_width, bias=False)
        # Xavier init for stable starting point
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, old_features: torch.Tensor) -> torch.Tensor:
        return self.proj(old_features)


def main(
    load_config: Path,
    new_resolution: int,
    new_nn_width: int,
    out_path: Optional[Path] = None,
    progress_indicator: Literal['tqdm', 'text'] = 'text',
    distill_steps: int = 1500,
    enable_e2e_distill: bool = True,
    e2e_distill_every: int = 50,
    e2e_distill_weight: float = 0.1,
    backbone_feature_weight: float = 1.0,
    temporal_coherence_weight: float = 0.5,
    temporal_coherence_n_times: int = 16,
    lr: float = 1e-2,
):
    """Distill coarse growth-aware field into fine one.

    Args:
        load_config: Path to config.yml
        new_resolution: New B-spline control points per dimension
        new_nn_width: New MLP width for backbone / weight_nn
        out_path: Output checkpoint path
        distill_steps: Number of optimization steps
        enable_e2e_distill: End-to-end ODE trajectory matching
        e2e_distill_every: E2E loss frequency
        e2e_distill_weight: E2E loss weight
        backbone_feature_weight: Feature matching weight (shared only)
        temporal_coherence_weight: Temporal coherence weight (shared only)
        temporal_coherence_n_times: Time points for coherence
        lr: Initial learning rate
    """
    config = yaml.load(load_config.read_text(), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)
    load_dir = config.get_checkpoint_dir()

    try:
        ckpt_path = max(load_dir.glob('*.ckpt'))
    except ValueError:
        raise ValueError(f'No checkpoint found in {load_dir}')

    old_config = config.pipeline.model.deformation_field
    old_field, key_map, pipeline_prefix = load_growth_field(ckpt_path, old_config)
    print(f'Pipeline prefix: {pipeline_prefix}')

    # New config
    new_config = copy.deepcopy(old_config)
    new_config.num_control_points = (new_resolution,) * 3
    new_config.weight_nn_width = new_nn_width
    if new_config.growth_num_control_points is not None:
        new_config.growth_num_control_points = (new_resolution,) * 3
    new_config.growth_nn_width = new_nn_width

    new_field = new_config.setup()

    # Architecture validation
    if _has_shared_encoder(old_field) != _has_shared_encoder(new_field):
        raise ValueError('Cannot distill across shared/independent architectures')

    # Detect backbone width mismatch
    use_shared = _has_shared_encoder(old_field)
    has_growth_field = _has_growth(old_field)
    old_bb_width = _get_backbone_width(old_field)
    new_bb_width = _get_backbone_width(new_field)
    width_mismatch = (
        use_shared
        and old_bb_width is not None
        and new_bb_width is not None
        and old_bb_width != new_bb_width
    )

    print(f'Shared encoder: {use_shared}')
    print(f'Growth enabled: {has_growth_field}')
    if use_shared:
        print(f'Backbone width: {old_bb_width} -> {new_bb_width} '
              f'({"MISMATCH - using projector" if width_mismatch else "same"})')

    # Move to GPU
    old_field = old_field.to('cuda')
    new_field = new_field.to('cuda')
    old_field.eval()

    # Feature projector (only when widths differ)
    feat_projector = None
    if width_mismatch:
        feat_projector = FeatureProjector(old_bb_width, new_bb_width).to('cuda')
        print(f'Created feature projector: {old_bb_width} -> {new_bb_width}')

    # Optimizer includes new_field params + projector params (if any)
    optim_params = list(new_field.parameters())
    if feat_projector is not None:
        optim_params += list(feat_projector.parameters())

    optimizer = torch.optim.AdamW(optim_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=distill_steps, eta_min=lr * 0.01
    )
    losses = []

    nq = new_resolution + 1
    phase2_start = int(distill_steps * 0.3)

    print(f'\nDistillation plan: {distill_steps} steps')
    print(f'  Phase 1 (0-{phase2_start}): field output + backbone feature matching')
    print(f'  Phase 2 ({phase2_start}-{distill_steps}): + e2e + temporal coherence')

    if progress_indicator == 'tqdm':
        pbar = trange(distill_steps)
    else:
        pbar = range(distill_steps)
        print('Optimizing: ', end='')

    for i in pbar:
        optimizer.zero_grad()

        # Sample spatial query points
        x = torch.linspace(-1, 1, nq)
        y = torch.linspace(-1, 1, nq)
        z = torch.linspace(-1, 1, nq)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        pos = torch.stack(
            [X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1
        ).to('cuda')

        t = torch.rand(1).to('cuda')
        x0, x1, x2 = pos[:, 0], pos[:, 1], pos[:, 2]

        # ============================================================
        # Loss 1: Velocity output matching
        # ============================================================
        with torch.no_grad():
            u_old = old_field.velocity(x0, x1, x2, t)
        u_new = new_field.velocity(x0, x1, x2, t)
        velocity_loss = F.mse_loss(u_old, u_new)

        # ============================================================
        # Loss 2: Growth output matching
        # ============================================================
        growth_loss = torch.zeros(1, device='cuda')
        if has_growth_field:
            with torch.no_grad():
                g_old = old_field.growth_rate(x0, x1, x2, t)
            g_new = new_field.growth_rate(x0, x1, x2, t)
            growth_loss = F.mse_loss(g_old, g_new)

        loss = velocity_loss + growth_loss

        # ============================================================
        # Loss 3: Backbone feature matching (shared encoder only)
        # When widths differ: project old features to new space first
        # ============================================================
        backbone_loss = torch.zeros(1, device='cuda')
        if use_shared and backbone_feature_weight > 0:
            with torch.no_grad():
                feat_old = old_field.shared_backbone(t.view(-1, 1))
            feat_new = new_field.shared_backbone(t.view(-1, 1))

            if width_mismatch and feat_projector is not None:
                # Project old features to new feature space
                feat_old_projected = feat_projector(feat_old)
                backbone_loss = F.mse_loss(feat_old_projected, feat_new)
            else:
                # Same width: direct MSE
                backbone_loss = F.mse_loss(feat_old, feat_new)

            bw = backbone_feature_weight if i < phase2_start else backbone_feature_weight * 0.3
            loss = loss + bw * backbone_loss

        # ============================================================
        # Loss 4: Temporal coherence (shared encoder, phase 2)
        # ============================================================
        temporal_loss = torch.zeros(1, device='cuda')
        if (
            use_shared
            and temporal_coherence_weight > 0
            and i >= phase2_start
            and i % 5 == 0
        ):
            t_grid = torch.linspace(0, 1, temporal_coherence_n_times,
                                    device='cuda').view(-1, 1)
            with torch.no_grad():
                feat_old_grid = old_field.shared_backbone(t_grid)
            feat_new_grid = new_field.shared_backbone(t_grid)

            if width_mismatch and feat_projector is not None:
                feat_old_grid_proj = feat_projector(feat_old_grid)
                temporal_loss = F.mse_loss(feat_old_grid_proj, feat_new_grid)
            else:
                temporal_loss = F.mse_loss(feat_old_grid, feat_new_grid)

            loss = loss + temporal_coherence_weight * temporal_loss

        # ============================================================
        # Loss 5: End-to-end ODE trajectory matching (phase 2)
        # ============================================================
        e2e_loss = torch.zeros(1, device='cuda')
        if (
            enable_e2e_distill
            and i >= phase2_start
            and i % e2e_distill_every == 0
        ):
            n_e2e = min(512, pos.shape[0])
            idx = torch.randperm(pos.shape[0])[:n_e2e]
            e2e_pos = pos[idx]
            t_start = torch.rand(1, device='cuda') * 0.5
            t_times = t_start.expand(n_e2e)

            with torch.no_grad():
                result_old = old_field(e2e_pos, t_times, 1.0)
            result_new = new_field(e2e_pos, t_times, 1.0)

            if isinstance(result_old, tuple) and isinstance(result_new, tuple):
                e2e_loss = (
                    F.mse_loss(result_old[0], result_new[0])
                    + F.mse_loss(result_old[1], result_new[1])
                )
            elif isinstance(result_old, tuple):
                e2e_loss = F.mse_loss(result_old[0], result_new)
            else:
                e2e_loss = F.mse_loss(result_old, result_new)

            loss = loss + e2e_distill_weight * e2e_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(optim_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if progress_indicator == 'tqdm':
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'v': f'{velocity_loss.item():.6f}',
                'g': f'{growth_loss.item():.6f}',
                'bb': f'{backbone_loss.item():.6f}',
                'tc': f'{temporal_loss.item():.6f}',
                'e2e': f'{e2e_loss.item():.6f}',
            })
        else:
            if i % 10 == 0:
                print('.', end='')
    print()

    # ================================================================
    # Visualization
    # ================================================================
    with torch.no_grad():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for idx, t_val in enumerate(np.linspace(0, 1, 6)):
            z = torch.linspace(-1, 1, 50).to('cuda')
            pos_viz = torch.stack(
                [torch.zeros_like(z), torch.zeros_like(z), z], dim=1
            )
            time_t = torch.ones_like(z) * t_val

            u_old_viz = old_field(pos_viz, time_t, 1.0)
            if isinstance(u_old_viz, tuple):
                u_old_viz = u_old_viz[0]
            u_old_viz = u_old_viz - pos_viz

            u_new_viz = new_field(pos_viz, time_t, 1.0)
            if isinstance(u_new_viz, tuple):
                u_new_viz = u_new_viz[0]
            u_new_viz = u_new_viz - pos_viz

            axes[0].plot(z.cpu(), u_old_viz[:, 2].cpu(),
                        label=f't={t_val:.2f}', ls='--', color=f'C{idx}')
            axes[0].plot(z.cpu(), u_new_viz[:, 2].cpu(), color=f'C{idx}')

        axes[0].set_xlabel('z')
        axes[0].set_ylabel('velocity_z')
        axes[0].set_title('Velocity (dashed=old, solid=new)')
        axes[0].legend(fontsize=8)

        if has_growth_field:
            for idx, t_val in enumerate(np.linspace(0, 1, 6)):
                z = torch.linspace(-1, 1, 50).to('cuda')
                x0_v = torch.zeros_like(z)
                t_t = torch.tensor(t_val, device='cuda').float()

                g_old_viz = old_field.growth_rate(x0_v, x0_v, z, t_t)
                g_new_viz = new_field.growth_rate(x0_v, x0_v, z, t_t)

                axes[1].plot(z.cpu(), g_old_viz.cpu(),
                            label=f't={t_val:.2f}', ls='--', color=f'C{idx}')
                axes[1].plot(z.cpu(), g_new_viz.cpu(), color=f'C{idx}')

            axes[1].set_xlabel('z')
            axes[1].set_ylabel('growth_rate')
            axes[1].set_title('Growth (dashed=old, solid=new)')
            axes[1].legend(fontsize=8)
        else:
            axes[1].text(0.5, 0.5, 'Growth disabled',
                        ha='center', va='center', transform=axes[1].transAxes)

        # Loss curve inset
        ax_loss = fig.add_axes([0.15, 0.65, 0.18, 0.2])
        ax_loss.semilogy(losses, 'k-', linewidth=0.5)
        ax_loss.set_xlabel('step', fontsize=7)
        ax_loss.set_ylabel('loss', fontsize=7)
        ax_loss.tick_params(labelsize=6)

        plt.tight_layout()
        fig_path = ckpt_path.parent.parent / 'growth_field_refining.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f'Visualization: {fig_path}')

    # ================================================================
    # Save checkpoint
    # ================================================================
    data = torch.load(ckpt_path, weights_only=False)
    new_dict = new_field.state_dict()

    updated, added, removed = 0, 0, 0

    for short_key, full_key in key_map.items():
        if short_key in new_dict:
            data['pipeline'][full_key] = new_dict[short_key].to('cuda')
            updated += 1
        else:
            del data['pipeline'][full_key]
            removed += 1
            print(f'  Removed: {full_key}')

    for short_key in new_dict:
        if short_key not in key_map:
            full_key = pipeline_prefix + short_key
            data['pipeline'][full_key] = new_dict[short_key].to('cuda')
            added += 1
            print(f'  Added: {full_key}')

    print(f'Checkpoint: updated={updated}, added={added}, removed={removed}')

    if out_path is None:
        out_path = ckpt_path.with_name(ckpt_path.stem + '-mod.ckpt')
    torch.save(data, out_path)
    print(f'Saved: {out_path}')

    # Quality report
    print('\n=== Distillation quality ===')
    print(f'Final loss: {losses[-1]:.6f}, min: {min(losses):.6f} @ step {np.argmin(losses)}')
    with torch.no_grad():
        for label, eval_fn in [('Velocity', lambda f, x0, x1, x2, t: f.velocity(x0, x1, x2, t)),
                                ('Growth', lambda f, x0, x1, x2, t: f.growth_rate(x0, x1, x2, t) if has_growth_field else None)]:
            devs = []
            for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
                t_q = torch.tensor(t_val, device='cuda')
                z = torch.linspace(-1, 1, 100, device='cuda')
                x0q = torch.zeros_like(z)
                out_old = eval_fn(old_field, x0q, x0q, z, t_q)
                out_new = eval_fn(new_field, x0q, x0q, z, t_q)
                if out_old is not None and out_new is not None:
                    devs.append(F.mse_loss(out_old, out_new).item())
            if devs:
                print(f'{label} MSE @ t=[0,.25,.5,.75,1]: {[f"{d:.2e}" for d in devs]}')


if __name__ == '__main__':
    import tyro
    tyro.cli(main)