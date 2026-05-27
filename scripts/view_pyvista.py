import numpy as np
import pyvista as pv

print("1. 正在加载数据...")
data = np.load('/home/dh524/data/dendrite_tiff/clean_volumes_2groups_v5/group2/frame_01620/volume_dendrite_enhanced.npy')
# vol = data[data.files[0]].astype(np.float32) # 解包時用
vol = data.astype(np.float32) # 不解包時用
if vol.ndim == 4: vol = vol[..., 0]

print("2. 几何对齐...")
# vol = vol.transpose(2, 0, 1)[::-1, ::-1, :] # dendrite_xray合成的三維結構用這個
vol = vol[::-1].copy() # 原始tiff輸出用這個

print("3. 数据截断与归一化...")
vmax = np.percentile(vol, 99.9)
if vmax == 0: vmax = 1.0 # 防御性编程，避免除以 0
vol = np.clip(vol / vmax, 0, 1) * 255.0
vol = vol.astype(np.uint8)

# 打印一下数值，确认是不是全变成 0 了
print(f"   -> [Debug] 数据统计: 最小={vol.min()}, 最大={vol.max()}, 平均={vol.mean():.2f}")

print("4. 构建 VTK 空间...")
nz, ny, nx = vol.shape
grid = pv.ImageData()
grid.dimensions = (nx, ny, nz)
# grid.spacing = (756.0/528.0, 756.0/528.0, 280.0/528.0) # dendrite_xray輸出用這個
grid.spacing = (1.0, 1.0, 1.0)  # 各向同性 原始tiff輸出用這個

# 使用更底层的字典赋值法，确保 PyVista 绝对能抓到数据
grid["density"] = vol.flatten(order="C")
grid.set_active_scalars("density")

print("5. 启动基础版渲染引擎...")
plotter = pv.Plotter()
plotter.set_background('black')

# 回退到最稳健的保底渲染参数
plotter.add_volume(
    grid,
    cmap="bone",
    opacity="linear",       # 换回自带的线性透明，确保所有东西都显示
    shade=False,            # 【关键】关掉光影计算，防止 CPU 渲染出死黑
    mapper="fixed_point"    # 强制使用兼容性最强的软件体素渲染器
)

# 添加一个外边框，防止黑天黑地失去方向感
plotter.add_bounding_box(color='white')

print("\n渲染就绪！")
plotter.show()