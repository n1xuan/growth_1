from typing import Optional
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from nerf_xray.deformation_fields import BSplineField3d, BSplineField1d, BsplineTemporalDeformationField3d, BsplineTemporalDeformationField3dConfig
from tqdm import tqdm, trange
import tyro

def load_def_field(p: Path, old_ng: int, weight_nn_width: int):
    print(f'Loading from {p}')
    data = torch.load(Path(p))
    _data = {}
    key_map = {}
    for key in data['pipeline'].keys():
        if 'deformation' in key:
            _data[key.split('deformation_field.')[1]] = data['pipeline'][key]
            key_map[key.split('deformation_field.')[1]] = key
    data = _data

    deformation_field = make_def_field(old_ng, weight_nn_width)
    deformation_field.load_state_dict(data)
    return deformation_field, key_map

def make_def_field(ng: int, weight_nn_width: int):
    config = BsplineTemporalDeformationField3dConfig(
        support_range=[(-1,1),(-1,1),(-1,1)],
        num_control_points=(ng,ng,ng),
        weight_nn_width=weight_nn_width
    )
    df2 = BsplineTemporalDeformationField3d(
        config=config
    )
    return df2

def main(
    ckpt_path: Path,
    old_resolution: int,
    new_resolution: int,
    old_nn_width: int,
    new_nn_width: int,
    out_path: Optional[Path] = None
):
    old_df, key_map = load_def_field(ckpt_path, old_resolution, old_nn_width)
    new_df = make_def_field(new_resolution, new_nn_width)
    # send to cuda
    old_df = old_df.to('cuda')
    new_df = new_df.to('cuda')

    optimizer = torch.optim.AdamW(new_df.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1.0, 0.01, 1000)
    losses = []
    pbar = trange(1000)

    for i in pbar:
        optimizer.zero_grad()
        nq = new_resolution+1
        x = torch.linspace(-1, 1, nq)
        y = torch.linspace(-1, 1, nq)
        z = torch.linspace(-1, 1, nq)
        X,Y,Z = torch.meshgrid(x,y,z, indexing='ij')
        pos = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1).to('cuda')
        _t = (i%20)/20
        t = _t*torch.ones_like(pos[:,0])
        uA = old_df(pos, t)
        uB = new_df(pos, t)
        loss = torch.nn.functional.mse_loss(uA, uB)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        pbar.set_postfix({'loss':loss.item(), 'lr':scheduler.get_last_lr()[0]})

    with torch.no_grad():
        for i,t in enumerate(np.linspace(0,1,6)):
            z = torch.linspace(-1, 1, 50).to('cuda')
            pos = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], dim=1).to('cuda')
            time = torch.ones_like(z)*t
            u = old_df(pos, time) - pos
            plt.plot(z.cpu(), u[:,2].cpu(), label=f'{t:.2f}', ls='--', color=f'C{i}')
            u = new_df(pos, time) - pos
            plt.plot(z.cpu(), u[:,2].cpu(), label=f'{t:.2f}', color=f'C{i}')
    plt.savefig(ckpt_path.with_name('def_field_refining.png'))
    plt.close()

    data = torch.load(ckpt_path)
    new_dict = new_df.state_dict()
    for key in key_map:
        data['pipeline'][key_map[key]] = new_dict[key].to('cuda')
    if out_path is None:
        out_path = ckpt_path.with_name(ckpt_path.stem+'-mod.ckpt')
    torch.save(data, out_path)
    print(f'Modified checkpoint saved to: {out_path}')

if __name__=='__main__':
    tyro.cli(main)