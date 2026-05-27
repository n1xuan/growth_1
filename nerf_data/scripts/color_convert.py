import json
from pathlib import Path
from typing import Optional, Literal

import cv2 as cv
import tyro
from rich.progress import track


def main(
    folder: Path, 
    fname_pattern: str = '*.png', 
    out_folder: Optional[Path] = None
) -> None:
    """Convert images in a folder to grayscale.

    Args:
        folder (Path): Input folder containing images.
        fname_pattern (str, optional): Pattern for matching image filenames. Defaults to '*.png'.
        out_folder (Optional[Path], optional): Output folder. If None, overwrite original images. Defaults to None.
    """
    assert folder.is_dir()
    print(f'Input folder: {folder}')

    if out_folder is None:
        out_folder = folder
        print('Will overwrite the input images')
    print(f'Output folder: {out_folder}')
    out_folder.mkdir(exist_ok=True)
    
    files = list(folder.glob(fname_pattern))
    for fn in track(files, description='Converting images'):
        im = cv.imread(str(fn), cv.IMREAD_GRAYSCALE)
        if im is None:
            print(f'Could not read {fn}')
            continue
        cv.imwrite(str(out_folder / fn.name), im)

if __name__=='__main__':
    tyro.cli(main)