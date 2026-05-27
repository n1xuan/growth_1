# %%
from typing import Optional, Literal, Any, Callable
import cv2 as cv
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tyro
from enum import Enum
# %%
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

m_thresh_min = None
m_thresh_max = None

def load_image(fn: Path, greyscale_fn: Callable) -> np.ndarray:
    # load unchanged. If not color, convert to color
    img = cv.imread(str(fn), cv.IMREAD_UNCHANGED)
    img = greyscale_fn(img).astype(img.dtype)
    # flip left-right
    img = img[:, ::-1]
    if img.ndim == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    return img

def main(
        input_folder: Path,
        output_folder: Optional[Path] = None,
        thresh_min: Optional[float] = None,
        thresh_max: Optional[float] = None,
        dtype: Optional[DTYPES] = DTYPES.UINT8,
        out_fn_pattern: Optional[str] = None,
        transparency: Optional[bool] = False,
        greyscale_fn: Optional[str] = None
):
    assert input_folder.exists()
    if output_folder is None:
        output_folder = input_folder
    output_folder.mkdir(parents=True, exist_ok=True)
    files = {int(fn.stem.split('_')[-1]):fn for fn in input_folder.glob('*.tif')}
    nums = list(files.keys())
    print(f'Found {len(files)} tiff files in {input_folder}')

    global m_thresh_min, m_thresh_max, img
    if thresh_min is not None:
        m_thresh_min = thresh_min
    if thresh_max is not None:
        m_thresh_max = thresh_max

    if greyscale_fn is None:
        greyscale_fn = lambda x: x
    else:
        print(f'Using greyscale function: {greyscale_fn}')
        greyscale_fn = eval(greyscale_fn)

    dtype = dtype.value

    def on_trackbar_thresh_min(val):
        global m_thresh_min
        m_thresh_min = val
        update_image()

    def on_trackbar_thresh_max(val):
        global m_thresh_max
        m_thresh_max = val
        update_image()

    def on_trackbar_rot(val):
        global img
        try:
            idx = nums[val]
            img = load_image(files[idx], greyscale_fn)
        except KeyError:
            pass
        update_image()

    def update_image():
        global m_thresh_min, m_thresh_max, img
        img_clip = threshold_image_colormap(img, m_thresh_min, m_thresh_max, DTYPES.UINT8.value)
        cv.imshow('image', img_clip)

    def threshold_image_colormap(
            img: np.ndarray,
            thresh_min: float, thresh_max: float,
            dtype: Optional[Any] = None
    ) -> np.ndarray:
        if dtype is None:
            dtype = img.dtype
        max_val = np.iinfo(dtype).max
        img_clip = img.astype(np.float64)
        vals_below = img < thresh_min
        vals_above = img > thresh_max
        img_clip = np.clip(img, thresh_min, thresh_max)
        # rescale between min and max
        img_clip = (img_clip - thresh_min) / (thresh_max - thresh_min) * max_val
        img_clip = img_clip.astype(dtype)
        # apply colormap
        img_clip = cv.applyColorMap(img_clip, cv.COLORMAP_JET)
        g = 0.5*max_val
        img_clip[vals_below] = g
        img_clip[vals_above] = g
        return img_clip

    def threshold_one_image(
            img: np.ndarray, 
            thresh_min: float, thresh_max: float, 
            dtype: Optional[Any] = None
    ) -> np.ndarray:
        if dtype is None:
            dtype = img.dtype
        max_val = np.iinfo(dtype).max
        img_clip = img.astype(np.float64)
        img_clip = np.clip(img, thresh_min, thresh_max)
        # rescale between min and max
        img_clip = (img_clip - thresh_min) / (thresh_max - thresh_min) * max_val
        img_clip = img_clip.astype(dtype)
        if transparency:
            mask = np.any(img_clip==max_val, axis=-1)
            img_clip = cv.cvtColor(img_clip, cv.COLOR_BGR2BGRA)
            img_clip[mask, 3] = 0
        else:
            img_clip = cv.cvtColor(img_clip, cv.COLOR_BGR2GRAY)
        return img_clip

    if thresh_max is None or thresh_min is None:
        img = load_image(files[nums[0]], greyscale_fn)
        max_val = np.iinfo(img.dtype).max
        print(f'Input dtype {img.dtype}, max value {max_val}')
        if m_thresh_min is None:
            m_thresh_min = 0
        if m_thresh_max is None:
            m_thresh_max = max_val
        cv.namedWindow('image', cv.WINDOW_NORMAL)
        cv.createTrackbar('threshold_min', 'image', 0, max_val, on_trackbar_thresh_min)
        cv.createTrackbar('threshold_max', 'image', 0, max_val, on_trackbar_thresh_max)
        cv.createTrackbar('rotation', 'image', 0, len(nums)-1, on_trackbar_rot)
        cv.waitKey(50)
        # select rectangle for flat field value
        r = cv.selectROI('image', img)
        update_image()
        k = cv.waitKey(0)
        flat_field = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2]), 0].mean()
        flat_field = (flat_field - m_thresh_min) / (m_thresh_max - m_thresh_min)
        print(f'Flat field: {flat_field:.3f}')
        cv.destroyAllWindows()
        print(f'Chosen thresholds: {m_thresh_min}, {m_thresh_max}')

    tif_files = list(input_folder.glob('*.tif'))
    pbar = tqdm(tif_files, desc='Processing images')
    im_min = np.iinfo(dtype).max
    im_max = np.iinfo(dtype).min
    for fn in pbar:
        img = load_image(fn, greyscale_fn)
        img = threshold_one_image(img, m_thresh_min, m_thresh_max, dtype)
        if out_fn_pattern is None:
            out_fn = str((output_folder/fn.stem).with_suffix('.png'))
        else:
            img_ind = int(fn.stem.split('_')[-1])
            out_fn = str(output_folder / out_fn_pattern.format(img_ind))
        cv.imwrite(out_fn, img)
        im_max = max(im_max, img.max())
        im_min = min(im_min, img.min())
        pbar.set_postfix_str(f'min: {im_min}, max: {im_max}')

if __name__ == '__main__':
    tyro.cli(main)