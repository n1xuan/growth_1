"""
Export 4D dendrite reconstruction as tiff stacks.

This extends the original exporter.py with tiff output and time-series export.
The original exporter supports NPZ/raw format; dendrite data is typically
visualized in ImageJ or ParaView which prefer tiff stacks.

Usage:
    # Export full time series (100 volumes)
    python export_dendrite.py volume-sequence \
        --load-config outputs/dendrite/xray_vfield/vel_12/config.yml \
        --output-dir exports/dendrite_4d/ \
        --num-timesteps 20 \
        --resolution 256

    # Export single volume at t=0.5
    python export_dendrite.py single-volume \
        --load-config outputs/dendrite/xray_vfield/vel_12/config.yml \
        --output-dir exports/dendrite_t05/ \
        --time 0.5
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import tyro
from rich.progress import track
from typing_extensions import Annotated, Literal

from nerfstudio.models.base_model import Model
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE


def export_volume_as_tiff(volume: np.ndarray, output_path: Path,
                          dtype: str = 'uint16'):
    """
    Save 3D volume as multi-page tiff (ImageJ-compatible).

    Args:
        volume: 3D array
        output_path: output tiff path
        dtype: 'uint8', 'uint16', or 'float32'
    """
    import tifffile

    dtypes = {'uint8': np.uint8, 'uint16': np.uint16, 'float32': np.float32}
    out_dtype = dtypes[dtype]

    vol = volume.copy()
    if out_dtype in (np.uint8, np.uint16):
        vmax = vol.max()
        if vmax > 0:
            vol = vol / vmax * np.iinfo(out_dtype).max
        vol = np.clip(vol, 0, np.iinfo(out_dtype).max)
    vol = vol.astype(out_dtype)

    tifffile.imwrite(str(output_path), vol)


def reconstruct_volume(pipeline, resolution: int, time: float = 0.0,
                       which: str = 'mixed') -> np.ndarray:
    """
    Reconstruct 3D volume from trained model at given time.

    Uses pipeline.eval_along_plane (same as exporter.py ExportVolumeGrid).

    Args:
        pipeline: trained pipeline
        resolution: volume resolution (isotropic)
        time: time in [0, 1]
        which: 'forward', 'backward', or 'mixed'

    Returns:
        volume: (resolution, resolution, resolution)
    """
    distances = np.linspace(-1, 1, resolution)
    slices = []
    for i_slice in range(resolution):
        xy = pipeline.eval_along_plane(
            plane='xy', distance=distances[i_slice],
            engine='numpy', resolution=resolution,
            target='field', time=time, which=which
        )
        slices.append(xy.squeeze())
    return np.stack(slices, axis=2)


@dataclass
class ExportVolumeSequence:
    """Export full time-series as tiff stack sequence."""

    load_config: Path
    """Path to config YAML file."""
    output_dir: Path
    """Output directory for tiff files."""
    resolution: int = 256
    """Volume resolution."""
    num_timesteps: int = 20
    """Number of time steps to export."""
    which: Literal['forward', 'backward', 'mixed'] = 'mixed'
    """Which model to use."""
    export_dtype: Literal['uint8', 'uint16', 'float32'] = 'uint16'
    """Output data type."""
    max_density: Optional[float] = None
    """Max density for normalization (None = auto-scan)."""

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)
        pipeline.model.eval()

        times = np.linspace(0, 1, self.num_timesteps)

        if self.max_density is None:
            CONSOLE.print('Scanning for global max density...')
            global_max = 0.0
            for t in [0.0, 0.5, 1.0]:
                vol = reconstruct_volume(pipeline, self.resolution // 2, t, self.which)
                global_max = max(global_max, vol.max())
            self.max_density = global_max
            CONSOLE.print(f'  Global max: {global_max:.4f}')

        for i, t in enumerate(track(times, description='Exporting volumes')):
            vol = reconstruct_volume(pipeline, self.resolution, t, self.which)
            fname = self.output_dir / f'volume_t{i:04d}.tiff'
            export_volume_as_tiff(vol, fname, self.export_dtype)

        meta = {
            'num_timesteps': self.num_timesteps,
            'resolution': self.resolution,
            'which': self.which,
            'max_density': float(self.max_density),
            'times': times.tolist(),
        }
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)

        CONSOLE.print(f':white_check_mark: Exported {self.num_timesteps} volumes '
                      f'to {self.output_dir}')


@dataclass
class ExportSingleVolume:
    """Export single volume at specific time."""

    load_config: Path
    """Path to config YAML file."""
    output_dir: Path
    """Output directory."""
    time: float = 0.0
    """Time to evaluate at."""
    resolution: int = 256
    """Volume resolution."""
    which: Literal['forward', 'backward', 'mixed'] = 'mixed'
    """Which model to use."""
    export_dtype: Literal['uint8', 'uint16', 'float32'] = 'uint16'
    """Output data type."""
    fmt: Literal['tiff', 'npz'] = 'tiff'
    """Output format."""

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)
        pipeline.model.eval()

        vol = reconstruct_volume(pipeline, self.resolution, self.time, self.which)
        print(f'Volume shape: {vol.shape}, range: [{vol.min():.4f}, {vol.max():.4f}]')

        if self.fmt == 'tiff':
            fname = self.output_dir / f'volume_t{self.time:.3f}.tiff'
            export_volume_as_tiff(vol, fname, self.export_dtype)
        else:
            fname = self.output_dir / f'volume_t{self.time:.3f}.npz'
            np.savez_compressed(str(fname), vol=vol)

        CONSOLE.print(f':white_check_mark: Exported to {fname}')


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ExportVolumeSequence, tyro.conf.subcommand(name="volume-sequence")],
        Annotated[ExportSingleVolume, tyro.conf.subcommand(name="single-volume")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()
