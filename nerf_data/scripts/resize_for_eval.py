import json
from pathlib import Path
from typing import Optional

import cv2 as cv
import tyro
from rich.progress import track


def main(folder: Path, fname_pattern: str = 'eval_*.png', downscale_factor: int = 3, out_folder: Optional[Path] = None):
    assert folder.is_dir()
    print(f'Input folder: {folder}')

    if out_folder is None:
        out_folder = Path(str(folder) + f'_{downscale_factor}')
    print(f'Output folder: {out_folder}')
    out_folder.mkdir(exist_ok=True)
    
    files = list(folder.glob(fname_pattern))
    for fn in track(files, description='Resizing images'):
        im = cv.imread(str(fn), cv.IMREAD_UNCHANGED)
        if im is None:
            print(f'Could not read {fn}')
            continue
        im = cv.resize(im, dsize=None, fx=1/downscale_factor, fy=1/downscale_factor, interpolation=cv.INTER_AREA)
        cv.imwrite(str(out_folder / fn.name), im)

if __name__=='__main__':
    tyro.cli(main)