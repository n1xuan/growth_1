"""
Volume grid utilities for dendrite data.

In neural_xray, volumetric supervision uses YAML files describing analytic
geometry (sphere collections, see balls_00.yaml). For experimental data
(lattice compression), the XCT-reconstructed volumes serve as ground truth.

For dendrite data, we have tiff stack volumes at each timestep. This module
converts them into the NPZ + YAML format needed by the training pipeline,
specifically by the datamanager's volume_grid_file parameter.

Note: nerf_xray/objects.py needs a new branch to load 'volume_grid' type.
See voxel_grid_object.py for the required Object subclass.
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Optional, Tuple


def load_tiff_volume(tiff_path: Path) -> np.ndarray:
    """
    Read a 3D volume from tiff file(s).

    Supports:
        - Single multi-page tiff file (e.g. volume.tiff)
        - Directory containing a single multi-page tiff (e.g. frame_01/volume.tif)
        - Directory of single-slice tiff files (e.g. slice_000.tif, ...)

    Args:
        tiff_path: path to tiff file or directory

    Returns:
        volume: 3D array, shape (Nz, Ny, Nx)
    """
    import tifffile

    tiff_path = Path(tiff_path)

    if tiff_path.is_file():
        volume = tifffile.imread(str(tiff_path))
    elif tiff_path.is_dir():
        files = sorted(tiff_path.glob('*.tif*'))
        assert len(files) > 0, f'No tiff files found in {tiff_path}'
        if len(files) == 1:
            # Single multi-page tiff inside a directory
            volume = tifffile.imread(str(files[0]))
        else:
            # Multiple single-slice tiff files
            slices = [tifffile.imread(str(f)) for f in files]
            volume = np.stack(slices, axis=0)
    else:
        raise FileNotFoundError(f'Not found: {tiff_path}')

    return volume.astype(np.float32)


def normalize_volume(volume: np.ndarray,
                     clip_percentile: Tuple[float, float] = (0.1, 99.9),
                     skip_if_normalized: bool = True) -> np.ndarray:
    """
    Normalize volume to [0, 1] with percentile clipping.

    For dendrite data: liquid background → low, solid dendrite → high.

    If the volume is already in [0, 1] range (as from volume_dendrite_enhanced),
    normalization is skipped to avoid distorting the data.

    Args:
        volume: raw 3D array
        clip_percentile: (low, high) percentiles for robust normalization
        skip_if_normalized: if True, skip when data already in [0, 1]

    Returns:
        normalized volume in [0, 1]
    """
    if skip_if_normalized and volume.min() >= 0.0 and volume.max() <= 1.0:
        return volume

    lo, hi = np.percentile(volume, clip_percentile)
    volume = np.clip(volume, lo, hi)
    volume = (volume - lo) / (hi - lo + 1e-8)
    return volume


def save_volume_grid(volume: np.ndarray, yaml_path: Path,
                     scene_extent: float = 2.0):
    """
    Save volume as NPZ + YAML pair for volumetric supervision.

    The YAML descriptor replaces the analytic geometry YAML used in
    neural_xray synthetic data (e.g. balls_00.yaml with sphere collections).

    Args:
        volume: normalized 3D array [0,1], shape (Nz, Ny, Nx)
        yaml_path: output path for YAML descriptor
        scene_extent: scene maps to [-extent/2, extent/2]^3
    """
    npz_path = yaml_path.with_suffix('.npz')
    np.savez_compressed(str(npz_path), volume=volume)

    config = {
        'type': 'volume_grid',
        'file': npz_path.name,
        'shape': list(volume.shape),
        'extent': float(scene_extent),
        'voxel_size': float(scene_extent / max(volume.shape)),
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f'✓ Saved volume grid: {yaml_path} + {npz_path}')
    print(f'  shape={volume.shape}, extent={scene_extent}, '
          f'voxel_size={config["voxel_size"]:.4f}')


def load_volume_grid(yaml_path: Path) -> Tuple[np.ndarray, dict]:
    """
    Load a volume grid from YAML + NPZ pair.

    Args:
        yaml_path: path to YAML descriptor

    Returns:
        (volume, config_dict)
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    assert config['type'] == 'volume_grid', \
        f'Expected type=volume_grid, got {config["type"]}'

    npz_path = yaml_path.parent / config['file']
    data = np.load(str(npz_path))
    volume = data['volume']

    return volume, config


def compute_volume_statistics(volume: np.ndarray) -> dict:
    """
    Compute basic statistics. Useful for sanity checks during data prep.

    Args:
        volume: 3D array

    Returns:
        dict with min, max, mean, std, nonzero_fraction
    """
    return {
        'min': float(volume.min()),
        'max': float(volume.max()),
        'mean': float(volume.mean()),
        'std': float(volume.std()),
        'nonzero_fraction': float((volume > 0.01).mean()),
        'shape': list(volume.shape),
    }
