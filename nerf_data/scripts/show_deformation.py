import json
from pathlib import Path
from enum import Enum
from typing import Optional, Tuple
import tyro
import numpy as np
from scipy.interpolate import interpn
import cv2 as cv
from matplotlib import cm

def main(
    input: Path, 
    component: int
):
    assert input.is_file(), f'Input file {input} does not exist'
    if input.suffix == '.npz':
        data = np.load(input)
        assert len(data) == 1, f'Expected only one array in npz file but got {len(data)}'
        data = next(iter(data.values()))
    else:
        data = np.load(input)
    print(f'Loaded array of shape {data.shape} and type {data.dtype}')
    data = data[..., component]
    assert data.ndim == 3, f'Expected 3D data but got {data.ndim}D data'
    # Show 3 slices, one from each axis. Create sliders to change the slice. Also add a slider to threshold the image
    _min = data.min()
    _max = data.max()
    _range = _max - _min
    def on_change_x(val):
        x = cv.getTrackbarPos('x', 'slices')
        y = cv.getTrackbarPos('y', 'slices')
        z = cv.getTrackbarPos('z', 'slices')
        mmin = cv.getTrackbarPos('min', 'slices') * _range / 100 + _min
        mmax = cv.getTrackbarPos('max', 'slices') * _range / 100 + _min
        _data = np.clip(data, mmin, mmax)
        _data = (_data - mmin) / (mmax - mmin)
        imshow = np.hstack([
            _data[x,:,::-1].T,
            _data[:,y,::-1].T,
            _data[:,::-1,z].T,
        ])
        # add color based on colormap
        imshow = cm.RdBu(imshow)[...,:3]
        # need to convert from 0-1 to 0-255
        imshow = (imshow * 255).astype(np.uint8)
        cv.imshow('slices', imshow)
    cv.namedWindow('slices', cv.WINDOW_GUI_NORMAL)
    cv.createTrackbar('x', 'slices', 0, data.shape[0]-1, on_change_x)
    cv.createTrackbar('y', 'slices', 0, data.shape[1]-1, on_change_x)
    cv.createTrackbar('z', 'slices', 0, data.shape[2]-1, on_change_x)
    cv.createTrackbar('min', 'slices', 0, 100, on_change_x)
    cv.createTrackbar('max', 'slices', 100, 100, on_change_x)
    on_change_x(0)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__=='__main__':
    tyro.cli(main)