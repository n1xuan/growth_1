from typing import Dict
from pathlib import Path
import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from nerf_xray.deformation_fields import BSplineField3d, BSplineField1d, BsplineTemporalIntegratedVelocityField3dConfig
from tqdm import tqdm, trange
import tyro

def make_def_field(ng: int, weight_nn_width: int):
    df2 = BsplineTemporalIntegratedVelocityField3dConfig(
        support_range=[(-1,1),(-1,1),(-1,1)],
        num_control_points=(ng,ng,ng),
        weight_nn_width=weight_nn_width
    ).setup()
    return df2

def make_1d_field(ng: int):
    df = BSplineField1d(
            torch.nn.parameter.Parameter(torch.linspace(0,1,ng)), 
            support_outside=True, 
            support_range=(0,1)
        )
    return df

def load_def_fields(p: Path, old_ng: int, weight_nn_width: int):
    print(f'Loading from {p}')
    data = torch.load(Path(p))
    _data_f = {}
    _data_b = {}
    _data_a = {}
    for key in data['pipeline'].keys():
        if 'deformation_field' in key:
            name = '.'.join(key.split('.')[2:])
            _data_f[name] = data['pipeline'][key]
        if 'field_weighing' in key:
            name = '.'.join(key.split('.')[2:])
            _data_a[name] = data['pipeline'][key]

    deformation_field_f = make_def_field(old_ng, weight_nn_width)
    deformation_field_f.load_state_dict(_data_f)
    f1d = make_1d_field(10)
    f1d.load_state_dict(_data_a)
    return deformation_field_f, f1d

def main(ckpt: Path, res: int, nn_width: int):
    assert ckpt.exists()
    df, f1d = load_def_fields(ckpt, res, nn_width)
    t = torch.linspace(0,1,100)
    with torch.no_grad():
        a = f1d(t)
    alphas = torch.nn.functional.sigmoid(a)
    div = torch.nn.functional.mse_loss(t, alphas)
    print(div)
    plt.plot(t, alphas)
    i = torch.searchsorted(alphas, torch.tensor(0.5))
    tstar = t[i].item()
    print(tstar)
    plt.plot(t, alphas*(1-alphas))
    plt.plot(t, torch.sigmoid(50*(alphas*(1-alphas)-0.2)))
    # plt.plot(t, t)
    # plt.plot(t, torch.nn.functional.sigmoid(phi0))
    plt.axhline(0.2, c='0.5', lw=0.5)
    plt.axhline(0.5, c='0.5', lw=0.5)
    plt.axvline(tstar, c='0.5', lw=0.5)
    plt.ylabel(r'$\alpha$')
    plt.xlim(0,1)
    plt.ylim(0,1)
    out_fn = ckpt.with_name('plot.png')
    plt.savefig(out_fn)
    plt.close()

if __name__=='__main__':
    tyro.cli(main)
