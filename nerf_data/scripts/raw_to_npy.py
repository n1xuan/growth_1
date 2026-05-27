import json
from pathlib import Path
from enum import Enum
from typing import Optional, Tuple
import tyro
import numpy as np
from scipy.interpolate import interpn

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
    resolution: Tuple[int, int, int],
    dtype: Optional[DTYPES] = DTYPES.UINT8,
    output: Optional[Path] = None,
    out_resolution: Optional[Tuple[int, int, int]] = None,
    out_dtype: Optional[DTYPES] = None,
    thresholds: Optional[Tuple[float, float]] = None
):
    """Convert raw binary data to shaped numpy array

    Args:
        input (Path): Path to .raw file
        resolution (Tuple[int, int, int]): Voxel resolution of the input file
        dtype (Optional[DTYPES]): Input datatype. Defaults to DTYPES.UINT8.
        output (Optional[Path]): Output path. If not provided, will replace suffix with .npz
        out_resolution (Optional[Tuple[int, int, int]]): Output resolution. If not provided, will keep the same resolution as input.
        out_dtype (Optional[DTYPES]): Output datatype. If not provided, will keep the same datatype as input.
    """
    assert input.is_file(), f'Input file {input} does not exist'
    dtype = dtype.value
    data = np.fromfile(input, dtype=dtype)
    print(f'Loaded {data.size} elements of type {dtype}')
    print(f'Min: {data.min()}, Max: {data.max()}, Mean: {data.mean()}, Std: {data.std()}')
    vol = data.reshape([resolution[i] for i in [2,1,0]])
    vol = vol.swapaxes(0,2)
    if out_resolution is None:
        pass
    else:
        X, Y, Z = np.meshgrid(
            np.linspace(0, 1, out_resolution[0]),
            np.linspace(0, 1, out_resolution[1]),
            np.linspace(0, 1, out_resolution[2]),
            indexing='ij'
        )
        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        vol = interpn(
            (
                np.linspace(0, 1, vol.shape[0]),
                np.linspace(0, 1, vol.shape[1]),
                np.linspace(0, 1, vol.shape[2]),
            ),
            vol,
            pts,
            method='linear',
            bounds_error=True,
        )
        vol = vol.reshape(X.shape).astype(dtype)
        print(vol.shape)
    if thresholds is not None:
        low, high = thresholds
        minval = vol.min()
        maxval = vol.max()
        rng = maxval - minval
        vol = np.clip(vol, minval + low*rng, minval + high*rng)
    if out_dtype is not None:
        # stretch to new dtype
        vol = vol.astype(np.float32)
        vol = vol - vol.min()
        vol = vol / vol.max()
        vol = vol * np.iinfo(out_dtype.value).max
        vol = vol.astype(out_dtype.value)
    if output is None:
        output = input.with_suffix('.npz')
    if output.suffix == '.npz':
        np.savez_compressed(output, vol=vol)
    else:
        assert output.suffix == '.npy'
        np.save(output, vol)

if __name__=='__main__':
    tyro.cli(main)