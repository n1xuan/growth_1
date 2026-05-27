# %%
from typing import Optional, Literal, Any
import cv2 as cv
import numpy as np
from pathlib import Path
from rich.progress import track
import tyro
from enum import Enum
# %%
def load_image(fn: Path) -> np.ndarray:
    # load unchanged. If not color, convert to color
    img = cv.imread(str(fn), cv.IMREAD_UNCHANGED)
    if img.ndim > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img

def main(
        file: Path,
):
    assert file.exists()  
    img = load_image(file)
    max_val = np.iinfo(img.dtype).max
    print(f'Input dtype {img.dtype}, max value {max_val}. Shape {img.shape}')
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.waitKey(50)
    # select rectangle for flat field value
    r = cv.selectROI('image', img)
    flat_field = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])].mean()
    print(f'Flat field: {flat_field / np.iinfo(img.dtype).max:.3f}')
    cv.destroyAllWindows()

if __name__ == '__main__':
    tyro.cli(main)