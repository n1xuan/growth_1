import json
from pathlib import Path
from enum import Enum
from typing import Optional, Tuple
import tyro
import numpy as np
import matplotlib.pyplot as plt

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
    downsample: Optional[int] = None,
    ylog: Optional[bool] = False,
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
    print(f'Min: {data.min()}, Max: {data.max()}, Mean: {data.mean()}, Std: {data.std()}')
    if downsample is not None:
        data = data[::downsample, ::downsample, ::downsample]
        print(f'Downsampled to {data.shape}')

    plt.figure(figsize=(6,4))
    plt.hist(data.ravel(), bins=100, fc='none', ec='black', lw=1)
    plt.xlabel('Value')
    if ylog:
        plt.yscale('log')
    plt.show()

if __name__=='__main__':
    tyro.cli(main)