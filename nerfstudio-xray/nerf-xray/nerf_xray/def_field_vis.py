import numpy as np
import cv2 as cv
import torch
import tyro
from pathlib import Path
from typing import Iterable, Optional, Tuple
import yaml

import deformation_fields

def_fields = {
    'bspline': deformation_fields.BsplineDeformationField3d(phi_x=None, support_outside=True, support_range=[(-1,1),(-1,1),(-1,1)], num_control_points=(6,6,6))
}

def draw_graph(
    z: np.ndarray, uz: np.ndarray, 
    resolution: Optional[int]=256, 
    zlims: Optional[Tuple[float,float]]=None, 
    uzlims: Optional[Tuple[float,float]]=None,
    pad: int = 10,
    col: Optional[Tuple[int,int,int]]=(120,120,255),
    im: Optional[np.ndarray] = None
):
    if zlims is None:
        zlims = (z.min(), z.max())
    if uzlims is None:
        uzlims = (uz.min(), uz.max())
    H = resolution
    W = resolution//2
    h = H - 2*pad
    w = W - 2*pad
    uz0 = -uzlims[0]/(uzlims[1]-uzlims[0])
    z = (z-zlims[0])/(zlims[1]-zlims[0]) # 0 to 1
    uz = (uz-uzlims[0])/(uzlims[1]-uzlims[0])
    z = H - (pad + z*h)
    uz = pad + uz*w
    uz0 = int(pad + uz0*w)
    z = z.astype(int)
    uz = uz.astype(int)

    if im is None:
        im = 255*np.ones((H,W,3), dtype=np.uint8)
        # draw axes
        cv.line(im, (uz0, H-pad), (uz0, pad), (0,0,0), 1, cv.LINE_8) # vertical
        cv.line(im, (pad, H-pad), (W-pad, H-pad), (0,0,0), 1, cv.LINE_8)

    # draw graph
    for i in range(z.shape[0]-1):
        im = cv.line(im, (uz[i], z[i]), (uz[i+1], z[i+1]), col, 1, cv.LINE_AA)

    return im

class ExpressionDisplacement:
    uz_string: str

    def displacement(self, x, y, z, i):
        uz = eval(f'lambda x,y,z: {self.uz_string}')
        assert i==2
        return uz(x,y,z)

def main(
    folder: Path,
    uz_string: Optional[str] = None
):
    load_config = folder / 'config.yml'
    lines = load_config.read_text().splitlines()
    # config = yaml.safe_load(load_config.read_text())
    field_type = None
    for line in lines:
        line = line.strip()
        if line.startswith('deformation_field'):
            field_type = line.split(':')[-1].strip()
    assert field_type is not None
    field = def_fields[field_type]

    load_ckpt = list(folder.glob('**/*.ckpt'))
    assert len(load_ckpt)==1
    load_ckpt = load_ckpt[0]
    assert load_ckpt.exists()
    data = torch.load(load_ckpt, map_location='cpu')
    data = data['pipeline']
    for key in data:
        if 'deformation' in key:
            break

    if field_type=='bspline':
        print(f'Setting phi_x to {key}, shape={data[key].shape}')
        field.bspline_field.phi_x.data = data[key]
        # field.bspline_field.phi_x.data = field.bspline_field.phi_x.data.permute(3,1,2,0)

    true_disp = None
    if uz_string is not None:
        true_disp = ExpressionDisplacement()
        true_disp.uz_string = uz_string

    cv.namedWindow('ux', cv.WINDOW_KEEPRATIO)
    cv.namedWindow('uy', cv.WINDOW_KEEPRATIO)
    cv.namedWindow('uz', cv.WINDOW_KEEPRATIO)


    def update_disp(val):
        x = (cv.getTrackbarPos('x','control')-50)/50.0
        y = (cv.getTrackbarPos('y','control')-50)/50.0

        z = torch.linspace(-1,1,50)
        zn = z.numpy()
        x = x*torch.ones_like(z)
        y = y*torch.ones_like(z)
        with torch.no_grad():
            ux = field.bspline_field.displacement(x,y,z,0).numpy()
            uy = field.bspline_field.displacement(x,y,z,1).numpy()
            uz = field.bspline_field.displacement(x,y,z,2).numpy()
        # ulims = (min(u.min() for u in [ux,uy,uz]), max(u.max() for u in [ux,uy,uz]))
        # print(ulims)
        ulims = (-0.1, 0.1)
        im_x = draw_graph(zn, ux, uzlims=ulims)
        im_y = draw_graph(zn, uy, uzlims=ulims)
        im_z = draw_graph(zn, uz, uzlims=ulims)
        if true_disp is not None:
            uz = true_disp.displacement(x,y,z,2).numpy()
            im_z = draw_graph(zn, uz, im=im_z, col=(100,100,100), uzlims=ulims)
        cv.imshow('ux', im_x)
        cv.imshow('uy', im_y)
        cv.imshow('uz', im_z)

    # control window
    cv.namedWindow('control', cv.WINDOW_FREERATIO)
    cv.createTrackbar('x', 'control', 0, 100, update_disp)
    cv.createTrackbar('y', 'control', 0, 100, update_disp)

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__=='__main__':
    tyro.cli(main)