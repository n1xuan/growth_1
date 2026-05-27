import json
from pathlib import Path
from enum import Enum
from typing import Optional, Tuple
import tyro
import numpy as np
from scipy.interpolate import interpn
import cv2 as cv

def main(
    input: Path, 
):
    assert input.is_file(), f'Input file {input} does not exist'
    if input.suffix == '.npz':
        data = np.load(input)['vol']
    else:
        data = np.load(input)
    print(f'Loaded {data.size} elements of type {data.dtype}')
    if data.dtype=='bool':
        data = 255*data.astype(np.uint8)
    # Show 3 slices, one from each axis. Create sliders to change the slice. Also add a slider to threshold the image
    def on_change(val):
        x = cv.getTrackbarPos('x', 'x-slice')
        y = cv.getTrackbarPos('y', 'y-slice')
        z = cv.getTrackbarPos('z', 'z-slice')
        mmin = cv.getTrackbarPos('min', 'x-slice')
        mmax = cv.getTrackbarPos('max', 'x-slice')
        _data = np.clip(data, mmin, mmax)
        _data = (_data - mmin) / (mmax - mmin)
        cv.imshow('x-slice', _data[x,:,:].T)
        cv.imshow('y-slice', _data[:,y,:].T)
        cv.imshow('z-slice', _data[:,:,z].T)
        
    cv.namedWindow('x-slice', cv.WINDOW_GUI_NORMAL)
    cv.namedWindow('y-slice', cv.WINDOW_GUI_NORMAL)
    cv.namedWindow('z-slice', cv.WINDOW_GUI_NORMAL)

    cv.createTrackbar('x', 'x-slice', 0, data.shape[0]-1, on_change)
    cv.createTrackbar('y', 'y-slice', 0, data.shape[1]-1, on_change)
    cv.createTrackbar('z', 'z-slice', 0, data.shape[2]-1, on_change)
    cv.createTrackbar('min', 'x-slice', 0, 255, on_change)
    cv.createTrackbar('max', 'x-slice', 255, 255, on_change)
    on_change(0)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__=='__main__':
    tyro.cli(main)