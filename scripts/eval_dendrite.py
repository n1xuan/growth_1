"""
Evaluate dendrite 4D reconstruction against ground truth tiff volumes.

This extends the original eval.py with ground truth volume comparison.
In neural_xray's experimental workflow, eval.py only computes projection-space
PSNR (comparing rendered vs measured radiographs). For dendrite data we have
ground truth 3D volumes at every timestep, enabling direct volumetric evaluation.

Usage:
    # Compare reconstructed volumes against ground truth tiffs
    python eval_dendrite.py compute-psnr-gt \
        --load-config outputs/dendrite/xray_vfield/vel_12/config.yml \
        --gt-dir /path/to/tiff_frames/ \
        --output-path results/dendrite_eval.json

    # Original projection-space PSNR (same as eval.py)
    python eval_dendrite.py compute-psnr \
        --load-config outputs/dendrite/xray_vfield/vel_12/config.yml \
        --output-path results/projection_psnr.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import tyro
from typing_extensions import Annotated, Literal
from rich.progress import track

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE

from volume_grid_utils import load_tiff_volume, normalize_volume


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """
    Compute reconstruction quality metrics between volumes.

    Args:
        pred: predicted volume, shape (D, H, W)
        target: ground truth volume, shape (D, H, W)

    Returns:
        dict with psnr, ssim, ncc, mse
    """
    assert pred.shape == target.shape, \
        f'Shape mismatch: {pred.shape} vs {target.shape}'

    pred = pred.astype(np.float64)
    target = target.astype(np.float64)

    # MSE and PSNR
    mse = np.mean((pred - target) ** 2)
    data_range = target.max() - target.min()
    if mse > 0 and data_range > 0:
        psnr = 10.0 * np.log10(data_range ** 2 / mse)
    else:
        psnr = float('inf')

    # Normalized cross-correlation
    pred_c = pred - pred.mean()
    target_c = target - target.mean()
    ncc = float(np.sum(pred_c * target_c) /
                (np.sqrt(np.sum(pred_c**2) * np.sum(target_c**2)) + 1e-12))

    # SSIM
    try:
        from skimage.metrics import structural_similarity
        ssim = structural_similarity(pred, target, data_range=data_range)
    except ImportError:
        ssim = None

    return {
        'psnr': float(psnr),
        'mse': float(mse),
        'ncc': float(ncc),
        'ssim': float(ssim) if ssim is not None else None,
    }


@dataclass
class ComputePSNRGroundTruth:
    """Evaluate reconstruction against ground truth tiff volumes."""

    load_config: Path
    """Path to config YAML file."""
    gt_dir: Path
    """Directory containing ground truth tiff frames."""
    output_path: Path = Path('output.json')
    """Output JSON file."""
    which: Literal['forward', 'backward', 'mixed'] = 'mixed'
    """Which model to evaluate."""
    resolution: int = 256
    """Volume resolution for export."""
    num_eval_frames: Optional[int] = None
    """Number of frames to evaluate (None = all)."""
    eval_every: int = 1
    """Evaluate every N-th frame."""

    def main(self) -> None:
        CONSOLE.print(f'[underline]Evaluating {self.load_config} '
                      f'against GT in {self.gt_dir}[/underline]')

        config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)
        pipeline.model.eval()

        gt_items = sorted([p for p in self.gt_dir.iterdir()
                           if p.suffix.lower() in ('.tif', '.tiff') or p.is_dir()])
        N = len(gt_items)
        if self.num_eval_frames is not None:
            N = min(N, self.num_eval_frames)

        eval_indices = list(range(0, N, self.eval_every))
        CONSOLE.print(f'Evaluating {len(eval_indices)} frames out of {N}')

        results = {}
        for i in track(eval_indices, description='Evaluating frames'):
            t = i / (N - 1) if N > 1 else 0.0

            # Reconstruct predicted volume at time t
            distances = np.linspace(-1, 1, self.resolution)
            pred_slices = []
            for i_slice in range(self.resolution):
                xy = pipeline.eval_along_plane(
                    plane='xy', distance=distances[i_slice],
                    engine='numpy', resolution=self.resolution,
                    target='field', time=t, which=self.which
                )
                pred_slices.append(xy.squeeze())
            pred_vol = np.stack(pred_slices, axis=2)

            # Load and resize ground truth
            gt_vol = load_tiff_volume(gt_items[i])
            gt_vol = normalize_volume(gt_vol)

            if gt_vol.shape != pred_vol.shape:
                from scipy.ndimage import zoom
                scales = [self.resolution / s for s in gt_vol.shape]
                gt_vol = zoom(gt_vol, scales, order=1)
                gt_vol = gt_vol[:self.resolution, :self.resolution, :self.resolution]

            metrics = compute_metrics(pred_vol, gt_vol)
            results[f't={t:.3f}'] = metrics
            CONSOLE.print(f'  t={t:.3f}: PSNR={metrics["psnr"]:.1f} dB, '
                          f'NCC={metrics["ncc"]:.3f}')

        # Summary statistics
        psnr_vals = [r['psnr'] for r in results.values() if r['psnr'] < float('inf')]
        ncc_vals = [r['ncc'] for r in results.values()]
        summary = {
            'mean_psnr': float(np.mean(psnr_vals)) if psnr_vals else None,
            'std_psnr': float(np.std(psnr_vals)) if psnr_vals else None,
            'mean_ncc': float(np.mean(ncc_vals)),
            'std_ncc': float(np.std(ncc_vals)),
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        benchmark_info = {
            'experiment_name': config.experiment_name,
            'method_name': config.method_name,
            'checkpoint': str(checkpoint_path),
            'which': self.which,
            'summary': summary,
            'per_frame': results,
        }
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), 'utf8')
        CONSOLE.print(f':white_check_mark: Saved results to: {self.output_path}')
        CONSOLE.print(f'  Mean PSNR: {summary["mean_psnr"]:.1f} ± '
                      f'{summary["std_psnr"]:.1f} dB')
        CONSOLE.print(f'  Mean NCC:  {summary["mean_ncc"]:.3f} ± '
                      f'{summary["std_ncc"]:.3f}')


@dataclass
class ComputeProjectionPSNR:
    """Projection-space PSNR evaluation (same as original eval.py)."""

    load_config: Path
    """Path to config YAML file."""
    output_path: Path = Path('output.json')
    """Output JSON file."""
    render_output_path: Optional[Path] = None
    """Optional path to save rendered images."""
    which: Literal['forward', 'backward', 'mixed'] = 'mixed'
    """Which model to evaluate."""

    def main(self) -> None:
        CONSOLE.print(f'[underline]Evaluating {self.load_config} '
                      f'in {self.which} mode[/underline]')
        config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)
        assert self.output_path.suffix == '.json'
        if self.render_output_path is not None:
            self.render_output_path.mkdir(parents=True, exist_ok=True)
        metrics_dict = pipeline.get_average_eval_image_metrics(
            output_path=self.render_output_path, get_std=True, which=self.which
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        benchmark_info = {
            'experiment_name': config.experiment_name,
            'method_name': config.method_name,
            'checkpoint': str(checkpoint_path),
            'results': metrics_dict,
        }
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), 'utf8')
        CONSOLE.print(f':white_check_mark: Saved results to: {self.output_path}')


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ComputePSNRGroundTruth, tyro.conf.subcommand(name="compute-psnr-gt")],
        Annotated[ComputeProjectionPSNR, tyro.conf.subcommand(name="compute-psnr")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()
