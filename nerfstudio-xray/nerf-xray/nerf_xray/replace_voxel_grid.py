"""
Script to combine forward and backward checkpoints.

Usually run after training forward and backward models.
The resulting checkpoint will append '_f' and '_b' to the keys of the forward and backward models, respectively.
"""
from typing import Optional
from pathlib import Path
import torch
import tyro
from nerf_xray.objects import VoxelGrid

def main(
    ckpt: Optional[Path] = None, 
    voxel_grid: Optional[Path] = None,
    out_fn: Optional[Path] = None
):
    assert ckpt is not None
    assert ckpt.exists(), f'Checkpoint {ckpt} does not exist'
    assert voxel_grid is not None
    assert voxel_grid.exists(), f'Voxel grid {voxel_grid} does not exist'
    voxel_grid_data = torch.load(voxel_grid, weights_only=False)
    if out_fn is None:
        out_fn = ckpt.with_name('combined.ckpt')

    combined_state_dict = {'pipeline':{}}
    data = torch.load(ckpt, weights_only=False)
    for key, val in data.items():
        if key=='step':
            if key not in combined_state_dict:
                combined_state_dict[key] = val
        elif key=='pipeline':
            for kk in val.keys():
                itms = kk.split('.')
                if itms[1] not in ['field', 'deformation_field']:
                    combined_state_dict['pipeline'][kk] = val[kk]
                else:
                    combined_state_dict['pipeline'][kk] = val[kk]
    
    print(f'Saving modified checkpoint to {out_fn}')
    torch.save(combined_state_dict, out_fn)
    
if __name__ == '__main__':
    tyro.cli(main)