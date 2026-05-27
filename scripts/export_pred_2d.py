import numpy as np
from pathlib import Path
import re
from PIL import Image
import sys

# 导入正向投影引擎 (确保路径正确)
sys.path.append('/home/dh524/data/projects/dendrite_xray_v1/scripts')
from parallel_beam_projector import ParallelBeamProjector

def natural_keys(path_obj):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', path_obj.name)]

def process_volume(file_path):
    data = np.load(file_path)
    vol = data[data.files[0]].astype(np.float32)
    if vol.ndim == 4: vol = vol[..., 0]
    # 几何对齐逻辑 (Z,Y,X -> X,Y,Z 并处理镜像)
    vol = vol.transpose(2, 0, 1)[::-1, ::-1, :]
    return vol

# 新增了 vol_nx 参数
def save_as_xray(projection, output_path, global_max, vol_nx, attenuation_k=1.0, target_size=None):
    """Beer-Lambert 物理成像逻辑，并完美还原 NeRF 空间缩放陷阱"""
    
    # 【核心物理修正】：切除投影仪擅自加的 padding，提取代表全空间的真实投影区
    dw = projection.shape[1]
    if dw > vol_nx:
        offset = (dw - vol_nx) // 2
        proj_valid = projection[:, offset : offset + vol_nx]
    else:
        proj_valid = projection

    # 使用裁剪后的真实数据计算 Beer-Lambert 衰减
    proj = proj_valid / global_max
    attenuation = np.exp(-attenuation_k * proj)
    img_uint8 = np.clip(attenuation * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_uint8, mode='L')
    
    # 此时再使用最高质量的 LANCZOS 插值，将这部分真实数据拉伸回 GT 的物理尺寸
    if target_size is not None:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
    img.save(str(output_path))

def main():
    # 路径配置
    input_dir = Path('/home/dh524/data/projects/dendrite_xray_v1/outputs/dendrite/spatiotemporal_mix/vel_51')
    output_dir = Path('/home/dh524/data/projects/dendrite_xray_v1/outputs/dendrite/pred/vel_51/2d_xray')
    output_dir.mkdir(parents=True, exist_ok=True)

    # ====== 【核心修复】：动态读取你真实的 GT 图像尺寸 ======
    # 脚本会自动去找你生成好的真实数据，拿到最权威的宽高比例
    gt_path = Path('/home/dh524/data/projects/dendrite_xray_v1/data/dendrite/images_00/train_00.png')
    if gt_path.exists():
        with Image.open(gt_path) as gt_img:
            physical_size = gt_img.size  # 获取真实的 (width, height)
        print(f"📐 成功读取真实 GT 物理尺寸: {physical_size}")
    else:
        print("⚠️ 警告：找不到 GT 图片，将使用回退尺寸 756x280")
        physical_size = (756, 280)
    # ======================================================

    # 1. 发现并排序所有帧
    pred_files = sorted(input_dir.glob('volume_frame*.npz'), key=natural_keys)
    if not pred_files:
        print("找不到输入文件，请检查路径。")
        return

    # 【修复点】：显式定义两个正交角度（完全对应 prepare_dendrite_data.py 的默认设置）
    target_angles = [0.0, 90.0]
    print(f"设定投影角度: {target_angles}")

    # 2. 第一轮遍历：收集所有预测，并计算全局最大值以保持对比度一致
    print("正在计算双角度下的物理射线积分与全局最大值...")
    all_raw_projs = [] # 结构: [每帧数据...], 内部为形状 (2, H, W) 的多角度张量
    global_max = 0.0
    
    for file_path in pred_files:
        vol = process_volume(file_path)
        projector = ParallelBeamProjector(vol, voxel_size=1.0)
        
        # project_batch 支持同时传入多个角度，直接返回 (2, Nz, det_width) 的张量
        projs = projector.project_batch(target_angles)

        # 将体积的 X 轴原始长度 (vol_nx) 随数据一起保存，用于后续精确裁剪
        vol_nx = vol.shape[2]
        all_raw_projs.append((projs, vol_nx))
        global_max = max(global_max, projs.max())

    # 3. 第二轮遍历：应用物理模型衰减，并分别保存两张图
    print(f"\n开始导出 2D X射线双角度图到: {output_dir}")
    
    # 修复1：加上 vol_nx 的解包
    for i, (projs, vol_nx) in enumerate(all_raw_projs):
        # 1. 改为：按照【每一帧】创建一个专属的子文件夹
        frame_dir = output_dir / f"frame_{i:02d}"
        frame_dir.mkdir(exist_ok=True)
        
        for angle_idx, angle in enumerate(target_angles):
            # 2. 图片名称改为对应的角度
            fname = f"{int(angle):03d}deg.png"
            
            # 修复2：把 vol_nx 传给 save_as_xray 激活裁剪逻辑
            save_as_xray(projs[angle_idx], frame_dir / fname, global_max, vol_nx, attenuation_k=1.0, target_size=physical_size)
            
        print(f"  已保存第 {i:02d} 帧的正交视角")

    print(f"\n全部 {len(pred_files) * 2} 张 2D X射线投影图导出完美结束！")

if __name__ == "__main__":
    main()