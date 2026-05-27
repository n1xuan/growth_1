import json
from pathlib import Path
from enum import Enum
from typing import Optional, Tuple
import tyro
import numpy as np
from scipy.interpolate import interpn
import cv2 as cv

class DTYPES(Enum):
    UINT8 = np.uint8
    UINT16 = np.uint16
    UINT32 = np.uint32
    UINT64 = np.uint64
    INT8 = np.int8
    INT16 = np.int16
    INT32 = np.int32
    INT64 = np.int64
    FLOAT32 = np.float32
    FLOAT64 = np.float64

def main(
    input: Path,
    resolution: Optional[Tuple[int, int, int]] = None,
    dtype: Optional[DTYPES] = None,
    downsample: Optional[int] = None
):
    assert input.is_file(), f'Input file {input} does not exist'
    if input.suffix == '.npz':
        data = np.load(input)['vol']
    elif input.suffix == '.raw':
        assert resolution is not None, 'Resolution must be provided for raw files'
        assert dtype is not None, 'Data type must be provided for raw files'
        dtype = dtype.value
        data = np.fromfile(input, dtype=dtype)
        assert len(data) == resolution[0] * resolution[1] * resolution[2], f'Expected {resolution[0] * resolution[1] * resolution[2]} elements but got {len(data)}'
        data = data.reshape(resolution)
        # swap XZ axes
        data = data.swapaxes(0,2)
    else:
        data = np.load(input)
    print(f'Loaded {data.size} elements of type {data.dtype}')
    if downsample is not None:
        data = data[::downsample, ::downsample, ::downsample]
        print(f'Downsampled to {data.shape}')
    data_min = data.min()
    data_max = data.max()
    data_range = data_max - data_min
    # Show 3 slices, one from each axis. Create sliders to change the slice. Also add a slider to threshold the image
    def on_change_x(val):
        x = cv.getTrackbarPos('x', 'slices')
        y = cv.getTrackbarPos('y', 'slices')
        z = cv.getTrackbarPos('z', 'slices')
        rng = data_max - data_min
        mmin = data_min + cv.getTrackbarPos('min(%)', 'slices') * 0.01 * data_range
        mmax = data_min + cv.getTrackbarPos('max(%)', 'slices') * 0.01 * data_range
        _data = np.clip(data, mmin, mmax)
        rng = max(1, mmax - mmin)
        _data = (_data - mmin) / rng
        imshow = np.hstack([
            _data[x,:,::-1].T,
            _data[:,y,::-1].T,
            _data[:,::-1,z].T,
        ])
        cv.imshow('slices', imshow)
    cv.namedWindow('slices', cv.WINDOW_GUI_NORMAL)
    cv.createTrackbar('x', 'slices', 0, data.shape[0]-1, on_change_x)
    cv.createTrackbar('y', 'slices', 0, data.shape[1]-1, on_change_x)
    cv.createTrackbar('z', 'slices', 0, data.shape[2]-1, on_change_x)
    cv.createTrackbar('min(%)', 'slices', 0, 100, on_change_x)
    cv.createTrackbar('max(%)', 'slices', 100, 100, on_change_x)
    on_change_x(0)

    cv.setTrackbarPos('x', 'slices', (data.shape[0]-1)//2)
    cv.setTrackbarPos('y', 'slices', (data.shape[1]-1)//2)
    cv.setTrackbarPos('z', 'slices', (data.shape[2]-1)//2)
    on_change_x(0)

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__=='__main__':
    tyro.cli(main)