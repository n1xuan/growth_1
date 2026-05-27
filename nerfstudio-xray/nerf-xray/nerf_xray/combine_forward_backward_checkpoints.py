"""
Script to combine forward and backward checkpoints.

Usually run after training forward and backward models.
The resulting checkpoint will append '_f' and '_b' to the keys of the forward and backward models, respectively.
"""
from typing import Optional
from pathlib import Path
import torch
import tyro

def main(
    fwd_ckpt: Optional[Path] = None, 
    bwd_ckpt: Optional[Path] = None, 
    out_fn: Optional[Path] = None
):
    assert fwd_ckpt is not None or bwd_ckpt is not None
    if fwd_ckpt is not None:
        assert fwd_ckpt.exists(), f'Forward checkpoint {fwd_ckpt} does not exist'
        if out_fn is None:
            out_fn = fwd_ckpt.with_name('combined.ckpt')
    if bwd_ckpt is not None:
        assert bwd_ckpt.exists(), f'Backward checkpoint {bwd_ckpt} does not exist'
        if out_fn is None:
            out_fn = bwd_ckpt.with_name('combined.ckpt')

    combined_state_dict = {'pipeline':{}}
    for direction, ckpt in zip(['f', 'b'], [fwd_ckpt, bwd_ckpt]):
        if ckpt is None: continue
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
                        itms[1] = itms[1] + '_' + direction
                        k = '.'.join(itms)
                        combined_state_dict['pipeline'][k] = val[kk]
                
    
    print(f'Saving modified checkpoint to {out_fn}')
    torch.save(combined_state_dict, out_fn)
    
if __name__ == '__main__':
    tyro.cli(main)