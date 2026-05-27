#!/usr/bin/env python3
"""
枝晶4D重建定量评估脚本

评估 neural_xray 导出的 3D volume 与 GT tiff 之间的重建质量。
处理尺寸不匹配（exporter 输出 528³，GT 是 280×528×528）和数值范围对齐。

用法:
    python eval_dendrite_4d.py \
        --gt-dir /home/dh524/data/dendrite_tiff/clean_volumes_2groups/merged_volumes_100 \
        --pred-dir outputs/dendrite/spatiotemporal_mix/vel_51 \
        --num-frames 20 \
        --output-path outputs/dendrite/spatiotemporal_mix/vel_51/eval_results.json
"""

import argparse
import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import zoom

import matplotlib
matplotlib.use('Agg')  # 无 GUI 环境
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import tifffile
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# 指标计算（不依赖 skimage，避免版本问题）
# ---------------------------------------------------------------------------

def compute_psnr(gt: np.ndarray, pred: np.ndarray, data_range: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio"""
    mse = np.mean((gt.astype(np.float64) - pred.astype(np.float64)) ** 2)
    if mse < 1e-15:
        return float('inf')
    return 10.0 * np.log10(data_range ** 2 / mse)


def compute_ssim_3d(gt: np.ndarray, pred: np.ndarray, data_range: float = 1.0) -> float:
    """
    Structural Similarity Index (逐切片计算取平均，避免 3D SSIM 的巨大内存开销)。
    沿 Z 轴逐切片计算 2D SSIM 后取平均。
    """
    try:
        from skimage.metrics import structural_similarity as _ssim
        ssim_vals = []
        for z in range(gt.shape[0]):
            s = _ssim(gt[z], pred[z], data_range=data_range)
            ssim_vals.append(s)
        return float(np.mean(ssim_vals))
    except ImportError:
        print("Warning: skimage not available, skipping SSIM")
        return float('nan')


def compute_correlation(gt: np.ndarray, pred: np.ndarray) -> float:
    """Pearson correlation coefficient"""
    corr, _ = pearsonr(gt.flatten(), pred.flatten())
    return float(corr)


def compute_mse(gt: np.ndarray, pred: np.ndarray) -> float:
    """Mean Squared Error"""
    return float(np.mean((gt.astype(np.float64) - pred.astype(np.float64)) ** 2))


# ---------------------------------------------------------------------------
# 数据加载与尺寸对齐
# ---------------------------------------------------------------------------

def load_gt_volume(tiff_path: Path) -> np.ndarray:
    """
    加载 GT tiff volume。
    支持单个 .tif 文件或包含 volume_dendrite_enhanced.tif 的目录。
    """
    tiff_path = Path(tiff_path)
    if tiff_path.is_dir():
        candidates = list(tiff_path.glob('volume_dendrite_enhanced.tif'))
        if not candidates:
            candidates = list(tiff_path.glob('*.tif'))
        assert len(candidates) > 0, f"No tiff files found in {tiff_path}"
        tiff_path = candidates[0]

    vol = tifffile.imread(str(tiff_path)).astype(np.float32)
    return vol


def load_pred_volume(npz_path: Path) -> np.ndarray:
    """
    加载 exporter 导出的 npz volume。
    自动检测 key 名称（'vol', 'density', 或第一个 key）。
    """
    data = np.load(str(npz_path))
    for key in ['vol', 'density', 'volume']:
        if key in data.files:
            vol = data[key].astype(np.float32)
            break
    else:
        vol = data[data.files[0]].astype(np.float32)

    if vol.ndim == 4:
        vol = vol[..., 0]

    return vol


def align_volumes(gt: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于绝对相机内参和最优轴序的 3D 几何对齐。
    修正了由于平行光投影 padding (756) 导致的空间畸变 (Anisotropic Scaling)。
    """
    print(f"    [Align] Original GT shape: {gt.shape}, Pred shape: {pred.shape}")
    
    # 1. 最优空间旋转和镜像翻转
    # transpose(2,0,1) 将 Exporter 输出的轴序修正回与 GT 匹配
    # [::-1, ::-1, :] 同时翻转了 Z 轴（修正 Coronal/Sagittal 上下颠倒）和 Y 轴（修正 Axial 镜像）
    p = pred.transpose(2, 0, 1)[::-1, ::-1, :]
    
    # 2. 几何变形校正 (复原真实的物理比例)
    # Z轴: 预测的 528 跨度代表真实的 280 高度 -> 需要压缩
    # Y/X轴: 预测的 528 跨度代表带 padding 的 756 宽度 -> 需要拉伸
    scale_z = 280 / 528.0
    scale_y = 756 / 528.0
    scale_x = 756 / 528.0
    
    print("    [Align] Resizing geometric space to (280, 756, 756)...")
    p_resized = zoom(p, (scale_z, scale_y, scale_x), order=1)
    
    # 3. 切除 X 和 Y 轴为了旋转而添加的黑边 (Padding)
    # 宽度从 756 恢复到 528：(756 - 528) = 228，两边各切除 114
    pad = 114
    p_final = p_resized[:, pad:-pad, pad:-pad]
    
    print(f"    [Align] Final aligned pred shape: {p_final.shape}")
    
    # 强检验：确保最终尺寸一模一样
    assert p_final.shape == gt.shape, f"Shape mismatch! GT: {gt.shape}, Pred: {p_final.shape}"
    
    return gt, p_final


def normalize_for_comparison(gt: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    数值范围对齐。

    GT: volume_dendrite_enhanced.tif, float32, [0, 1]
    Pred: exporter 输出的 density，float32，范围不确定

    策略：
    1. 先 clip pred 到非负
    2. 对 pred 做线性缩放使其 [0, max] 映射到 [0, 1]
       （因为 exporter 输出的密度值的绝对尺度与 GT 不同）
    3. GT 保持 [0, 1] 不变
    """
    gt = gt.copy()
    pred = pred.copy()

    # clip 负值
    pred = np.clip(pred, 0.0, None)

    # 使用99.9%分位数抗噪
    pred_max = np.percentile(pred, 99.9) 
    if pred_max > 0:
        pred = pred / pred_max

    # 超过 99.9% 分位数的极少数超亮噪点，直接强行截断到 1.0
    pred = np.clip(pred, 0.0, 1.0)

    # GT 也确保在 [0, 1]
    gt = np.clip(gt, 0.0, 1.0)

    return gt, pred


# ---------------------------------------------------------------------------
# 帧映射
# ---------------------------------------------------------------------------

def natural_keys(path_obj):
    """将字符串里的数字提出来按数值比大小"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', path_obj.name)]

def discover_frame_mapping(
    gt_dir: Path,
    pred_dir: Path,
    num_frames: int,
) -> List[Dict]:
    """
    自动发现 GT 帧和 pred 帧的对应关系。

    GT 目录结构: frame_01/, frame_02/, ..., frame_20/
    Pred 文件名: volume_frame00_t-0.npz, volume_frame01_t-0.05263.npz, ...
    """
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)

    # 发现 GT 帧目录
    gt_candidates = sorted([
        p for p in gt_dir.iterdir()
        if p.is_dir() and p.name.startswith('frame_')
    ], key=natural_keys)[:num_frames]

    # 发现 pred 文件
    pred_files = sorted(pred_dir.glob('volume_frame*_t-*.npz'), key=natural_keys)[:num_frames]

    mapping = []
    for i in range(min(num_frames, len(gt_candidates), len(pred_files))):
        t = i / (num_frames - 1) if num_frames > 1 else 0.0
        entry = {
            'frame_idx': i,
            'time': t,
            'gt_path': str(gt_candidates[i]),
            'pred_path': str(pred_files[i]) if i < len(pred_files) else None,
        }
        mapping.append(entry)

    return mapping


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def save_slice_comparison(
    gt: np.ndarray,
    pred: np.ndarray,
    frame_idx: int,
    time: float,
    output_path: Path,
):
    """保存 GT vs Pred 的三正交切片对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    z_mid = gt.shape[0] // 2
    y_mid = gt.shape[1] // 2
    x_mid = gt.shape[2] // 2

    vmin, vmax = 0, max(gt.max(), pred.max())

    titles = [
        f'Axial (z={z_mid})',
        f'Coronal (y={y_mid})',
        f'Sagittal (x={x_mid})',
    ]
    gt_slices = [gt[z_mid], gt[:, y_mid, :], gt[:, :, x_mid]]
    pred_slices = [pred[z_mid], pred[:, y_mid, :], pred[:, :, x_mid]]

    for col in range(3):
        axes[0, col].imshow(gt_slices[col], cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f'GT {titles[col]}')
        axes[0, col].axis('off')

        axes[1, col].imshow(pred_slices[col], cmap='gray', vmin=vmin, vmax=vmax)
        axes[1, col].set_title(f'Pred {titles[col]}')
        axes[1, col].axis('off')

    fig.suptitle(f'Frame {frame_idx:02d} (t={time:.3f})', fontsize=16)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()


def save_temporal_gif(
    pred_volumes: List[np.ndarray],
    times: List[float],
    output_path: Path,
    plane: str = 'axial',
):
    """生成时间序列 GIF 动画"""
    if plane == 'axial':
        slices = [v[v.shape[0] // 2] for v in pred_volumes]
    elif plane == 'coronal':
        slices = [v[:, v.shape[1] // 2, :] for v in pred_volumes]
    else:
        slices = [v[:, :, v.shape[2] // 2] for v in pred_volumes]

    vmax = max(s.max() for s in slices)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(slices[0], cmap='gray', vmin=0, vmax=vmax)
    ax.set_title(f't = {times[0]:.3f}')
    ax.axis('off')

    def update(i):
        im.set_data(slices[i])
        ax.set_title(f't = {times[i]:.3f}')
        return [im]

    anim = FuncAnimation(fig, update, frames=len(slices), interval=300)
    anim.save(str(output_path), writer='pillow', fps=3)
    plt.close()
    print(f"  Saved animation: {output_path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='枝晶4D重建定量评估')
    parser.add_argument('--gt-dir', type=str, required=True,
                        help='GT tiff 目录 (包含 frame_XX/ 子目录)')
    parser.add_argument('--pred-dir', type=str, required=True,
                        help='导出的 npz 文件目录')
    parser.add_argument('--num-frames', type=int, default=20,
                        help='评估帧数')
    parser.add_argument('--output-path', type=str, default=None,
                        help='评估结果 JSON 输出路径')
    parser.add_argument('--save-vis', action='store_true', default=True,
                        help='保存可视化图像')
    parser.add_argument('--save-gif', action='store_true', default=True,
                        help='保存时间序列 GIF')
    parser.add_argument('--skip-ssim', action='store_true', default=False,
                        help='跳过 SSIM 计算（3D SSIM 很慢）')
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    output_path = Path(args.output_path) if args.output_path else pred_dir / 'eval_results.json'

    # 1. 发现帧映射
    print('=' * 60)
    print('Step 1: Discovering frame mapping')
    print('=' * 60)
    mapping = discover_frame_mapping(gt_dir, pred_dir, args.num_frames)
    print(f"Found {len(mapping)} frame pairs")
    for m in mapping:
        print(f"  Frame {m['frame_idx']:02d}: t={m['time']:.4f}")
        print(f"    GT:   {m['gt_path']}")
        print(f"    Pred: {m['pred_path']}")

    # 2. 逐帧评估
    print('\n' + '=' * 60)
    print('Step 2: Evaluating frames')
    print('=' * 60)

    results = []
    pred_volumes_for_gif = []
    times_for_gif = []

    for entry in mapping:
        if entry['pred_path'] is None:
            print(f"\n  Frame {entry['frame_idx']:02d}: pred file missing, skipping")
            continue

        print(f"\n  Frame {entry['frame_idx']:02d} (t={entry['time']:.4f}):")

        # 加载
        gt = load_gt_volume(Path(entry['gt_path']))
        pred = load_pred_volume(Path(entry['pred_path']))

        print(f"    GT range:   [{gt.min():.4f}, {gt.max():.4f}], mean={gt.mean():.4f}")
        print(f"    Pred range: [{pred.min():.4f}, {pred.max():.4f}], mean={pred.mean():.4f}")

        # 尺寸对齐
        gt_aligned, pred_aligned = align_volumes(gt, pred)

        # 数值归一化
        gt_norm, pred_norm = normalize_for_comparison(gt_aligned, pred_aligned)

        # 计算指标
        metrics = {
            'frame_idx': entry['frame_idx'],
            'time': entry['time'],
            'psnr': compute_psnr(gt_norm, pred_norm, data_range=1.0),
            'correlation': compute_correlation(gt_norm, pred_norm),
            'mse': compute_mse(gt_norm, pred_norm),
        }

        if not args.skip_ssim:
            metrics['ssim'] = compute_ssim_3d(gt_norm, pred_norm, data_range=1.0)

        print(f"    PSNR:        {metrics['psnr']:.2f} dB")
        print(f"    Correlation: {metrics['correlation']:.4f}")
        print(f"    MSE:         {metrics['mse']:.6f}")
        if 'ssim' in metrics:
            print(f"    SSIM:        {metrics['ssim']:.4f}")

        results.append(metrics)

        # 可视化
        if args.save_vis:
            vis_dir = output_path.parent / 'visualizations'
            vis_dir.mkdir(parents=True, exist_ok=True)
            save_slice_comparison(
                gt_norm, pred_norm,
                entry['frame_idx'], entry['time'],
                vis_dir / f'comparison_frame{entry["frame_idx"]:02d}.png'
            )

        # 收集 GIF 数据
        if args.save_gif:
            pred_volumes_for_gif.append(pred_norm)
            times_for_gif.append(entry['time'])

    # 3. 汇总统计
    print('\n' + '=' * 60)
    print('Step 3: Summary')
    print('=' * 60)

    if results:
        avg_psnr = np.mean([r['psnr'] for r in results if np.isfinite(r['psnr'])])
        avg_corr = np.mean([r['correlation'] for r in results])
        avg_mse = np.mean([r['mse'] for r in results])

        print(f"\n  Average PSNR:        {avg_psnr:.2f} dB")
        print(f"  Average Correlation: {avg_corr:.4f}")
        print(f"  Average MSE:         {avg_mse:.6f}")

        if any('ssim' in r for r in results):
            avg_ssim = np.mean([r['ssim'] for r in results if 'ssim' in r])
            print(f"  Average SSIM:        {avg_ssim:.4f}")

        summary = {
            'num_frames': len(results),
            'avg_psnr': float(avg_psnr),
            'avg_correlation': float(avg_corr),
            'avg_mse': float(avg_mse),
        }
        if any('ssim' in r for r in results):
            summary['avg_ssim'] = float(avg_ssim)

        # 保存结果
        output = {'summary': summary, 'per_frame': results}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(output_path), 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to: {output_path}")

    # 4. 生成 GIF
    if args.save_gif and pred_volumes_for_gif:
        print('\n' + '=' * 60)
        print('Step 4: Generating animations')
        print('=' * 60)
        gif_dir = output_path.parent / 'visualizations'
        gif_dir.mkdir(parents=True, exist_ok=True)

        for plane in ['axial', 'coronal', 'sagittal']:
            save_temporal_gif(
                pred_volumes_for_gif, times_for_gif,
                gif_dir / f'dendrite_growth_{plane}.gif',
                plane=plane,
            )


if __name__ == '__main__':
    main()
