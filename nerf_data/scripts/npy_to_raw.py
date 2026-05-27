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
    output: Optional[Path] = None,
    out_resolution: Optional[Tuple[int, int, int]] = None,
    out_dtype: Optional[DTYPES] = None,
    thresholds: Optional[Tuple[float, float]] = None
):
    """Convert numpy array to raw binary data

    Args:
        input (Path): Path to .npy or .npz file
        dtype (Optional[DTYPES]): Output datatype. Defaults to DTYPES.UINT8.
        output (Optional[Path]): Output path. If not provided, will replace suffix with .raw
        out_resolution (Optional[Tuple[int, int, int]]): Output resolution. If not provided, will keep the same resolution as input.
        out_dtype (Optional[DTYPES]): Output datatype. If not provided, will keep the same datatype as input.
        thresholds (Optional[Tuple[float, float]]): Optional threshold values for clipping (low, high) as fraction of max value
    """
    assert input.is_file(), f'Input file {input} does not exist'
    
    # Load the numpy array
    if input.suffix == '.npz':
        vol = np.load(input)['vol']
    else:
        assert input.suffix == '.npy'
        vol = np.load(input)
    
    print(f'Loaded array of shape {vol.shape} with dtype {vol.dtype}')
    
    # Handle resolution change if needed
    if out_resolution is not None:
        orig_dtype = vol.dtype
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
        vol = vol.reshape(X.shape)
        vol = vol.astype(orig_dtype)
        print(f'Resampled to shape {vol.shape}')
    
    # Handle thresholds if provided
    if thresholds is not None:
        low, high = thresholds
        minval = vol.min()
        maxval = vol.max()
        rng = maxval - minval
        vol = np.clip(vol, minval + low*rng, minval + high*rng)
    
    # Handle dtype conversion if needed
    if out_dtype is not None:
        # stretch to new dtype
        vol = vol.astype(np.float32)
        vol = vol - vol.min()
        vol = vol / vol.max()
        vol = vol * np.iinfo(out_dtype.value).max
        vol = vol.astype(out_dtype.value)
    
    # Reorder axes to match raw format (Z,Y,X order) and flatten
    vol = vol.swapaxes(0, 2)
    vol = vol.ravel()  # This flattens the array in C-order (row-major)
    
    # Save to raw file
    if output is None:
        output = input.with_suffix('.raw')
    vol.tofile(output)
    print(f'Saved raw file to {output}')

if __name__=='__main__':
    tyro.cli(main) 