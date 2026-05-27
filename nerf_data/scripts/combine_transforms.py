import json
import os
import sys
from pathlib import Path
from typing import Optional
import tyro


def main(folder: Path, timestamp_func: Optional[str] = None, enforce_exists: bool = True):
    """Combine transform files from folder into a single file.

    Args:
        folder (Path): path to folder containing transform files.
            transform files are named transforms_*.json. 
        timestamp_func (Optional[str], optional): Function to convert timestamp to time.
            Use lambda function. If not provided, the timestamp is used as is.
    """
    if timestamp_func is not None:
        t_f = eval(timestamp_func)
    else:
        t_f = lambda x: float(x)
    assert folder.is_dir()
    transforms = None
    for fn in folder.glob('transforms_*.json'):
        timestamp = int(fn.stem.split('_')[-1])
        t = t_f(timestamp)
        print(fn)
        d = json.loads(fn.read_text())
        for f in d['frames']:
            # assert 'time' in f
            _t = f.get('time', t)
            assert _t==t, f"Expected time {t} but got {_t}"
            f['time'] = round(t,2)
        frames = []
        for frame in d['frames']:
            fn = frame['file_path']
            if enforce_exists and not (folder/fn).exists():
                print(f'File {fn} does not exist. Dropping frame')
                continue
            frames.append(frame)
        d['frames'] = frames
        if transforms is None:
            transforms = d
        else:
            transforms['frames'].extend(d['frames'])
    out_fname = folder/'transforms.json'
    print(f'Writing to {out_fname}')
    out_fname.write_text(json.dumps(transforms, indent=2))

if __name__=='__main__':
    tyro.cli(main)
