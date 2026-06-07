"""
Multi-resolution refinement for growth-aware velocity field.
CORRECTED VERSION — adds end-to-end distillation check, proper new key handling.

Extends refine_vfield.py to handle both velocity and growth B-spline fields
when upsampling from coarse to fine resolution.

The distillation process:
1. Load checkpoint with old (coarse) growth-aware field
2. Create new (fine) field with higher B-spline resolution
3. Optimize new field to match old field's velocity AND growth outputs
4. Optionally: end-to-end ODE trajectory matching every N steps
5. Save modified checkpoint

Usage:
    python refine_growth_vfield.py \
        --load-config outputs/.../config.yml \
        --new-resolution 15 \
        --new-nn-width 32

FIX vs original:
- Added end-to-end distillation loss (optional, every 50 steps)
- Added proper handling of new keys that don't exist in old checkpoint
- Added growth_nn_width update in new_config
"""
from typing import Optional, Literal
from pathlib import Path
import torch
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
    
    Extracts all keys containing 'deformation' from the pipeline state dict
    and loads them into a fresh field instance.
    """
    print(f'Loading from {ckpt_path}')
    data = torch.load(ckpt_path, weights_only=False)
    _data = {}
    key_map = {}
    for key in data['pipeline'].keys():
        if 'deformation' in key:
            short_key = key.split('deformation_field.')[1]
            _data[short_key] = data['pipeline'][key]
            key_map[short_key] = key
    
    field = old_config.setup()
    field.load_state_dict(_data)
    return field, key_map


def main(
    load_config: Path,
    new_resolution: int,
    new_nn_width: int,
    out_path: Optional[Path] = None,
    progress_indicator: Literal['tqdm', 'text'] = 'text',
    distill_steps: int = 1000,
    enable_e2e_distill: bool = True,
    e2e_distill_every: int = 50,
    e2e_distill_weight: float = 0.1,
):
    """Distill coarse growth-aware field into fine one.
    
    Args:
        load_config: Path to config.yml of the trained model
        new_resolution: New B-spline control points per dimension
        new_nn_width: New MLP width for weight_nn and growth_nn
        out_path: Output checkpoint path (default: input with -mod suffix)
        progress_indicator: 'tqdm' for interactive, 'text' for scripts
        distill_steps: Number of optimization steps
        enable_e2e_distill: Whether to add end-to-end ODE trajectory matching
        e2e_distill_every: Compute e2e loss every N steps
        e2e_distill_weight: Weight for e2e loss relative to field-level losses
    """
    config = yaml.load(load_config.read_text(), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)
    load_dir = config.get_checkpoint_dir()
    
    try:
        ckpt_path = max(load_dir.glob('*.ckpt'))
    except ValueError:
        raise ValueError(f'No checkpoint found in {load_dir}')
    
    print(f'Loading from {ckpt_path}')
    old_config = config.pipeline.model.deformation_field
    old_field, key_map = load_growth_field(ckpt_path, old_config)
    print(f'Old field config: {old_config}')
    
    # Create new higher-resolution config
    new_config = copy.deepcopy(old_config)
    new_config.num_control_points = (new_resolution, new_resolution, new_resolution)
    new_config.weight_nn_width = new_nn_width
    # Growth field resolution follows velocity field unless specified
    if new_config.growth_num_control_points is None:
        pass  # will default to num_control_points in __init__
    else:
        new_config.growth_num_control_points = (
            new_resolution, new_resolution, new_resolution
        )
    new_config.growth_nn_width = new_nn_width
    
    new_field = new_config.setup()
    print(f'New field config: {new_config}')
    
    # Move to GPU
    old_field = old_field.to('cuda')
    new_field = new_field.to('cuda')
    old_field.eval()
    
    optimizer = torch.optim.AdamW(new_field.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 1.0, 0.01, distill_steps
    )
    losses = []

    nq = new_resolution + 1
    
    if progress_indicator == 'tqdm':
        pbar = trange(distill_steps)
    else:
        pbar = range(distill_steps)
        print('Optimizing field: ', end='')

    for i in pbar:
        optimizer.zero_grad()
        
        x = torch.linspace(-1, 1, nq)
        y = torch.linspace(-1, 1, nq)
        z = torch.linspace(-1, 1, nq)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        pos = torch.stack(
            [X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1
        ).to('cuda')
        
        t = torch.rand(1).to('cuda')
        x0, x1, x2 = pos[:, 0], pos[:, 1], pos[:, 2]
        
        # --- Distill velocity field (instantaneous) ---
        with torch.no_grad():
            phi_old = old_field.weight_nn(t.view(-1, 1)).view(
                *old_field.bspline_field.grid_size, 3
            )
            u_old = old_field.disp_func(x0, x1, x2, phi_x=phi_old)
        
        phi_new = new_field.weight_nn(t.view(-1, 1)).view(
            *new_field.bspline_field.grid_size, 3
        )
        u_new = new_field.disp_func(x0, x1, x2, phi_x=phi_new)
        
        velocity_loss = F.mse_loss(u_old, u_new)
        
        # --- Distill growth field (instantaneous) ---
        growth_loss = torch.zeros(1, device='cuda')
        if (
            hasattr(old_field, 'growth_nn')
            and old_field.growth_nn is not None
            and hasattr(new_field, 'growth_nn')
            and new_field.growth_nn is not None
        ):
            with torch.no_grad():
                g_old = old_field.growth_rate(x0, x1, x2, t)
            g_new = new_field.growth_rate(x0, x1, x2, t)
            growth_loss = F.mse_loss(g_old, g_new)
        
        loss = velocity_loss + growth_loss
        
        # --- Optional: end-to-end ODE trajectory matching ---
        e2e_loss = torch.zeros(1, device='cuda')
        if enable_e2e_distill and i % e2e_distill_every == 0:
            # Use a subset of points for e2e (expensive)
            n_e2e = min(512, pos.shape[0])
            idx = torch.randperm(pos.shape[0])[:n_e2e]
            e2e_pos = pos[idx]
            t_start = torch.rand(1, device='cuda') * 0.5  # early time
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
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        
        if progress_indicator == 'tqdm':
            pbar.set_postfix({
                'loss': loss.item(),
                'v_loss': velocity_loss.item(),
                'g_loss': growth_loss.item(),
                'e2e': e2e_loss.item(),
                'lr': scheduler.get_last_lr()[0],
            })
        else:
            if i % 10 == 0:
                print('.', end='')
    print()

    # Visualization: velocity and growth along z-axis
    with torch.no_grad():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot velocity
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
            
            axes[0].plot(
                z.cpu(), u_old_viz[:, 2].cpu(),
                label=f't={t_val:.2f}', ls='--', color=f'C{idx}',
            )
            axes[0].plot(
                z.cpu(), u_new_viz[:, 2].cpu(), color=f'C{idx}',
            )
        axes[0].set_xlabel('z')
        axes[0].set_ylabel('velocity_z')
        axes[0].set_title('Velocity field (dashed=old, solid=new)')
        axes[0].legend()
        
        # Plot growth rate
        if new_field.growth_nn is not None:
            for idx, t_val in enumerate(np.linspace(0, 1, 6)):
                z = torch.linspace(-1, 1, 50).to('cuda')
                x0_v = torch.zeros_like(z)
                t_t = torch.tensor(t_val, device='cuda')
                # type transform
                x0_v, z, t_t = x0_v.float(), z.float(), t_t.float()
                
                g_old_viz = old_field.growth_rate(x0_v, x0_v, z, t_t)
                g_new_viz = new_field.growth_rate(x0_v, x0_v, z, t_t)
                
                axes[1].plot(
                    z.cpu(), g_old_viz.cpu(),
                    label=f't={t_val:.2f}', ls='--', color=f'C{idx}',
                )
                axes[1].plot(
                    z.cpu(), g_new_viz.cpu(), color=f'C{idx}',
                )
            axes[1].set_xlabel('z')
            axes[1].set_ylabel('growth_rate')
            axes[1].set_title('Growth field (dashed=old, solid=new)')
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(ckpt_path.parent.parent / 'growth_field_refining.png')
        plt.close()

    # Save refined checkpoint
    # FIX: handle new keys that may not exist in old checkpoint
    data = torch.load(ckpt_path, weights_only=False)
    new_dict = new_field.state_dict()
    
    # Update existing keys
    for key in key_map:
        if key in new_dict:
            data['pipeline'][key_map[key]] = new_dict[key].to('cuda')
        else:
            print(f'Warning: key {key} not in new field state_dict')
    
    # FIX: Add new keys that exist in new_dict but not in old checkpoint
    # This happens when growth field resolution changes and new B-spline buffers appear
    prefix = 'model.deformation_field.'
    for key in new_dict:
        full_key = f'_model.{prefix[len("model."):]}{key}'
        # Try both possible prefixes (with and without underscore for DDP)
        pipeline_key = f'model.deformation_field.{key}'
        if pipeline_key not in data['pipeline'] and key not in key_map:
            data['pipeline'][pipeline_key] = new_dict[key].to('cuda')
            print(f'Added new key: {pipeline_key}')
    
    if out_path is None:
        out_path = ckpt_path.with_name(ckpt_path.stem + '-mod.ckpt')
    torch.save(data, out_path)
    print(f'Modified checkpoint saved to: {out_path}')


if __name__ == '__main__':
    import tyro
    tyro.cli(main)
