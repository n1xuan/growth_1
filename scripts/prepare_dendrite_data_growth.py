#!/usr/bin/env python3
"""
Prepare dendrite tiff data for neural_xray training pipeline.

This is the dendrite equivalent of generate_data.py in neural_xray.

In neural_xray's *experimental* workflow (not the synthetic balls demo),
the input is:
    - High-fidelity XCT reconstructions at t=0 and t=T (first/last)
    - Sparse radiographs during deformation (1-2 per timestep)

For dendrite data, we have ground truth tiff volumes at every timestep,
so we can:
    - Use t=0 and t=T volumes directly as canonical volume supervision
    - Synthesize parallel beam projections from all timesteps

Key differences from neural_xray experimental workflow:
    - Parallel beam geometry (synchrotron) instead of cone beam (lab XCT)
    - Transform matrices use large R=1000 to approximate parallel beam
    - Volume grids stored as NPZ instead of analytic YAML geometry

Usage:
    python prepare_dendrite_data.py \
        --data-dir /path/to/tiff_frames/ \
        --output-dir data/dendrite \
        --num-frames 20 \
        --num-train-angles 16 \
        --num-intermediate-angles 2

Output structure matches neural_xray convention:
    data/dendrite/
    ├── transforms_00.json
    ├── transforms_19.json
    ├── transforms_00_to_19.json
    ├── dendrite_00.yaml + .npz
    ├── dendrite_19.yaml + .npz
    ├── images_00/ ... images_19/
"""

import sys
import json
import shutil
import numpy as np
from pathlib import Path
from typing import List, Optional

import tyro

from parallel_beam_projector import ParallelBeamProjector
from volume_grid_utils import (
    load_tiff_volume, normalize_volume, save_volume_grid, compute_volume_statistics
)


def build_parallel_beam_transform_matrix(theta_deg: float) -> list:
    """
    Build 4x4 camera-to-world transform matrix for parallel beam ORTHOPHOTO.

    For ORTHOPHOTO in nerfstudio:
        - All rays have direction = c2w @ [0, 0, -1] (same for all pixels)
        - Ray origins are placed on a grid in the c2w-transformed plane
        - The c2w translation column is overwritten per-pixel by nerfstudio,
          so we set it to zero here (it only matters that R is correct)

    Convention (same as neural_xray / nerfstudio):
        - Column 0: camera right (x)
        - Column 1: camera up (y)
        - Column 2: camera backward (-viewing direction) (z)
        - Column 3: camera position (set to 0; ORTHOPHOTO overrides per-pixel)

    For parallel beam at azimuthal angle theta (rotation around z-axis):
        - beam direction (into scene) = (cos_t, sin_t, 0)
        - camera -z = beam direction => camera z = (-cos_t, -sin_t, 0)
        - camera x = (-sin_t, cos_t, 0)  (perpendicular, in x-y plane)
        - camera y = (0, 0, 1)            (vertical, along z)

    Args:
        theta_deg: azimuthal rotation angle in degrees

    Returns:
        4x4 matrix as nested list
    """
    theta = np.radians(theta_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Camera plane distance from origin (must be > scene_extent/2 = 1.0)
    D = 3.0

    # beam_direction = (cos_t, sin_t, 0) = camera's -z direction
    # camera z-axis (backward) = (-cos_t, -sin_t, 0)
    # camera x-axis (right)    = (-sin_t, cos_t, 0)
    # camera y-axis (up)       = (0, 0, 1)
    # translation              = -D * beam_direction (place plane behind scene)
    tx = -D * cos_t
    ty = -D * sin_t
    tz = 0.0

    transform_matrix = [
        [-sin_t, 0.0, -cos_t, tx],
        [cos_t,  0.0, -sin_t, ty],
        [0.0,    1.0,  0.0,   tz],
        [0.0,    0.0,  0.0,   1.0],
    ]
    return transform_matrix


def compute_ortho_focal_lengths(image_height: int, image_width: int,
                                scene_extent: float) -> dict:
    """
    Compute focal length parameters for ORTHOPHOTO camera.

    In nerfstudio ORTHOPHOTO, the normalized pixel coordinate is:
        coord_x = (pixel_x - cx) / fx
        coord_y = (pixel_y - cy) / fy

    These become the ray origin positions (in camera-local units) before
    c2w transform. So fx controls the mapping: each pixel spans 1/fx
    world units horizontally. For the detector to cover scene_extent:

        fx = image_width / scene_extent
        fy = image_height / scene_extent

    We also store camera_angle_x for compatibility (though ORTHOPHOTO
    doesn't use it for ray generation, the dataparser may still read it).

    Args:
        image_height: projection image height in pixels
        image_width: projection image width in pixels
        scene_extent: physical scene width in world units

    Returns:
        dict with fl_x, fl_y, cx, cy, w, h, camera_angle_x
    """
    fl_x = image_width / scene_extent
    fl_y = image_height / scene_extent
    cx = image_width / 2.0
    cy = image_height / 2.0

    # camera_angle_x: not used by ORTHOPHOTO but may be read by dataparser
    # Set to a reasonable value for compatibility
    camera_angle_x = 2.0 * np.arctan(scene_extent / 2.0)

    return {
        'fl_x': float(fl_x),
        'fl_y': float(fl_y),
        'cx': float(cx),
        'cy': float(cy),
        'w': int(image_width),
        'h': int(image_height),
        'camera_angle_x': float(camera_angle_x),
    }


def save_projection_image(projection: np.ndarray, output_path: Path,
                          global_max: Optional[float] = None, attenuation_k:float = 0.5):
    """
     将 line integral 转为 Beer-Lambert attenuation 并保存为 8-bit 灰度 PNG。

    物理模型与 xray_renderer.py 中 AttenuationRenderer.forward 一致：
        attenuation = exp(-∫ density ds)
    
    AttenuationRenderer.forward (xray_renderer.py 第41-43行) 做的是：
        delta_density = ray_samples.deltas * densities
        acc = torch.sum(delta_density, dim=-2)   # line integral
        attenuation = torch.exp(-acc)             # Beer-Lambert

    所以模型预测的 rgb 值：
        背景 (acc=0)     → exp(0) = 1.0  → 白色 (255)
        物体 (acc=large) → exp(-x) ≈ 0   → 黑色 (0)
    GT图像必须匹配这个约定。

    Args:
        projection: 2D array，line integral 值 (≥0)
        output_path: 输出 PNG 路径
        global_max: 全局最大 line integral（跨所有帧一致归一化）
        attenuation_k: 对比度控制参数
            k=0.5 → 图像 range [155, 255]，与 Go 渲染器 balls 数据一致
            k=1.0 → 图像 range [94, 255]，更高对比度
            k=2.0 → 图像 range [35, 255]，非常高对比度
    """
    from PIL import Image

    proj = projection.copy()

    # step1:归一化
    pmax = global_max if global_max is not None else proj.max()
    if pmax > 0:
        # proj = proj / pmax * 255.0  # 线性映射：背景(0) → 0(黑)，物体(max) → 255(白)但 AttenuationRenderer 输出的是：背景 → 1.0(白)，物体 → ~0(黑)
        proj = proj / pmax
    
    # step2:Beer-Lambert转换，与 AttenuationRenderer.forward 一致
    # normalized_proj=0 (背景) → exp(0) = 1.0 (白)
    # normalized_proj=1 (最密) → exp(-k) (暗)

    attenuation = np.exp(-attenuation_k * proj)

    # step3:映射到8-bit
    img_uint8 = np.clip(attenuation * 255.0, 0, 255).astype(np.uint8) 
    img = Image.fromarray(img_uint8, mode='L')
    img.save(str(output_path))


def main(
    data_dir: Path,
    output_dir: Path = Path('data/dendrite'),
    num_frames: int = 20,
    num_train_angles: int = 16,
    num_intermediate_angles: int = 2,
    resolution: Optional[int] = None,
    scene_extent: float = 2.0,
    voxel_size: float = 1.0,
    downscale_factor: int = 1,
    verify_fbp: bool = False,
    attenuation_k: float = 0.5,
):
    """
    Prepare dendrite data for neural_xray training.

    Args:
        data_dir: directory containing tiff frames (sorted = time order)
        output_dir: output directory for prepared data
        num_frames: number of time frames to process
        num_train_angles: projection angles for first/last frames (full coverage)
        num_intermediate_angles: projection angles for intermediate frames (sparse)
        resolution: target projection resolution (None = native)
        scene_extent: scene range in normalized coordinates
        voxel_size: physical voxel size for line integral scaling
        downscale_factor: spatial downscale factor for volumes before projection
        verify_fbp: verify projector with FBP on first frame
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== Step 1: Discover tiff frames =====
    print('=' * 50)
    print('Step 1: Discovering tiff frames')
    print('=' * 50)

    data_dir = Path(data_dir)
    assert data_dir.exists(), f'Data directory not found: {data_dir}'

    # Find frames: support tiff files, or directories containing tiff files
    # (e.g. frame_01/volume_dendrite_enhanced.tif)
    candidates = sorted([p for p in data_dir.iterdir()
                         if p.suffix.lower() in ('.tif', '.tiff')
                         or (p.is_dir() and (p / 'volume_dendrite_enhanced.tif').exists())])
    assert len(candidates) >= num_frames, \
        f'Found {len(candidates)} tiff items, need {num_frames}'

    tiff_items = candidates[:num_frames]
    N = num_frames
    print(f'Found {len(tiff_items)} frames, using {N}')
    print(f'First: {tiff_items[0].name}')
    print(f'Last:  {tiff_items[N-1].name}')

    for i in range(N):
        (output_dir / f'images_{i:02d}').mkdir(exist_ok=True)

    # Angles: first/last get full coverage, intermediate get sparse
    train_angles = np.linspace(0, 180, num_train_angles, endpoint=False).tolist()
    intermediate_angles = np.linspace(0, 180, num_intermediate_angles,
                                       endpoint=False).tolist()

    print(f'\nTrain angles ({num_train_angles}): {train_angles[:4]}...')
    print(f'Intermediate angles ({num_intermediate_angles}): {intermediate_angles}')

    # ===== Step 2: Process first and last frames =====
    print('')
    print('=' * 50)
    print('Step 2: Processing first and last frames (volume grids)')
    print('=' * 50)

    print(f'\nLoading frame {N-1}: {tiff_items[N-1].name}')
    item_last = tiff_items[N-1]
    tiff_path_last = item_last / 'volume_dendrite_enhanced.tif' if item_last.is_dir() else item_last
    vol_last = load_tiff_volume(tiff_path_last)
    vol_last = normalize_volume(vol_last)
    stats = compute_volume_statistics(vol_last)
    print(f'  Stats: {stats}')

    save_volume_grid(vol_last, output_dir / f'dendrite_{N-1:02d}.yaml', scene_extent)

    if verify_fbp:
        print('\nVerifying projector with FBP round-trip...')
        projector = ParallelBeamProjector(vol_last, voxel_size)
        _, psnr = projector.verify_with_fbp(num_angles=180)
        print(f'  FBP verification PSNR: {psnr:.1f} dB')

    # ===== Step 3: Generate projections for all frames =====
    # Two-pass approach: first scan global max intensity, then save with
    # consistent normalization. This ensures relative intensities between
    # frames are preserved (critical for time-series reconstruction).
    print('')
    print('=' * 50)
    print('Step 3: Generating projections (pass 1: scan global max)')
    print('=' * 50)

    camera_angle_x = compute_ortho_focal_lengths  # placeholder, computed after first projection
    all_projections = {}  # i -> (projections_array, angles_list)
    global_proj_max = 0.0
    ortho_params = None  # will be set from first projection's shape

    for i in range(N):
        t = i / (N - 1)
        item = tiff_items[i]
        tiff_path = item / 'volume_dendrite_enhanced.tif' if item.is_dir() else item
        vol = load_tiff_volume(tiff_path)
        vol = normalize_volume(vol)

        if downscale_factor > 1:
            from scipy.ndimage import zoom
            vol = zoom(vol, 1.0 / downscale_factor, order=1)

        if i == N - 1:
            angles = train_angles
        else:
            angles = intermediate_angles

        projector = ParallelBeamProjector(vol, voxel_size)
        projections = projector.project_batch(angles)
        all_projections[i] = (projections, angles)

        # Compute ORTHOPHOTO focal lengths from first projection shape
        if ortho_params is None:
            proj_h, proj_w = projections.shape[1], projections.shape[2]
            ortho_params = compute_ortho_focal_lengths(proj_h, proj_w, scene_extent)
            print(f'  ORTHOPHOTO params: fl_x={ortho_params["fl_x"]:.1f}, '
                  f'fl_y={ortho_params["fl_y"]:.1f}, '
                  f'image=({proj_h}, {proj_w})')

        frame_max = projections.max()
        global_proj_max = max(global_proj_max, frame_max)
        print(f'  Frame {i:02d}: {len(angles)} proj, max={frame_max:.4f}')

    print(f'\n  Global projection max: {global_proj_max:.4f}')

    print('')
    print('=' * 50)
    print('Step 3b: Saving projections (pass 2: consistent normalization)')
    print('=' * 50)

    all_transforms = {}

    for i in range(N):
        t = i / (N - 1)
        projections, angles = all_projections[i]

        image_dir = output_dir / f'images_{i:02d}'
        frames = []
        for ia, angle in enumerate(angles):
            fname = f'train_{ia:02d}.png'
            save_projection_image(projections[ia], image_dir / fname,
                                  global_max=global_proj_max, attenuation_k = attenuation_k)

            frame = {
                'file_path': f'images_{i:02d}/{fname}',
                'transform_matrix': build_parallel_beam_transform_matrix(angle),
                'time': t,
            }
            frames.append(frame)

        # Create eval image (copy first train image, same as neural_xray convention)
        eval_src = image_dir / 'train_00.png'
        eval_dst = image_dir / 'eval_00.png'
        if eval_src.exists():
            shutil.copy2(eval_src, eval_dst)

        if frames:
            eval_frame = frames[0].copy()
            eval_frame['file_path'] = f'images_{i:02d}/eval_00.png'
            frames.append(eval_frame)

        all_transforms[i] = frames
        print(f'  ✓ Frame {i:02d}: saved {len(angles)} images')

    # ===== Step 4: Assemble transforms JSON files =====
    print('')
    print('=' * 50)
    print('Step 4: Assembling transform files')
    print('=' * 50)

    base_meta = {
        'camera_model': 'ORTHOPHOTO',
        'camera_angle_x': ortho_params['camera_angle_x'],
        'fl_x': ortho_params['fl_x'],
        'fl_y': ortho_params['fl_y'],
        'cx': ortho_params['cx'],
        'cy': ortho_params['cy'],
        'w': ortho_params['w'],
        'h': ortho_params['h'],
        'scene_extent': scene_extent,
    }

    # transforms_00.json (first frame, full angles)
    t00 = {**base_meta, 'frames': all_transforms[0]}
    t00_path = output_dir / 'transforms_00.json'
    with open(t00_path, 'w') as f:
        json.dump(t00, f, indent=2)
    print(f'✓ Created {t00_path} ({len(all_transforms[0])} frames)')

    # transforms_{N-1}.json (last frame, full angles)
    tN = {**base_meta, 'frames': all_transforms[N-1]}
    tN_path = output_dir / f'transforms_{N-1:02d}.json'
    with open(tN_path, 'w') as f:
        json.dump(tN, f, indent=2)
    print(f'✓ Created {tN_path} ({len(all_transforms[N-1])} frames)')

    # transforms_00_to_{N-1}.json (all frames aggregated)
    all_frames = []
    for i in range(N):
        all_frames.extend(all_transforms[i])

    t_all = {**base_meta, 'frames': all_frames}
    t_all_path = output_dir / f'transforms_00_to_{N-1:02d}.json'
    with open(t_all_path, 'w') as f:
        json.dump(t_all, f, indent=2)
    print(f'✓ Created {t_all_path} ({len(all_frames)} frames)')

    # ===== Summary =====
    print('')
    print('=' * 50)
    print('✓ Data preparation complete!')
    print('=' * 50)
    print(f'  Output directory: {output_dir}')
    print(f'  Frames: {N}')
    print(f'  Transform files:')
    print(f'    - {t00_path.name}')
    print(f'    - {tN_path.name}')
    print(f'    - {t_all_path.name}')
    print(f'  Volume grids:')
    print(f'    - dendrite_{N-1:02d}.yaml/npz')
    print(f'  Image directories: images_00/ ... images_{N-1:02d}/')


if __name__ == '__main__':
    tyro.cli(main)
