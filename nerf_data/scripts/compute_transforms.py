import os
import math
import json
from pathlib import Path
from typing import Optional, Tuple, Literal, Union
import pandas as pd
import numpy as np
import tyro
from scipy.spatial.transform import Rotation

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        row_list = []
        for col in row:
            row_list.append(round(col, 10))
        matrix_list.append(row_list)
    return matrix_list

def load_xtekct(fn: Path):
    if isinstance(fn, str): fn = Path(fn)
    if fn.is_file():
        pth = fn
    else:
        pth = next(fn.glob('*.xtekct'))
    assert pth.exists()
    txt = pth.read_text()
    lines = txt.split('\n')
    print(f'Loaded {len(lines)} from "{pth}"')
    data = {}

    for line in lines:
        if ('[' in line) and (']' in line):
            current = line.strip('[]')
            data[current] = {}
        elif len(line)<1:
            pass
        else:
            fields = line.split('=')
            key = fields[0]
            value = '='.join(fields[1:])
            try:
                value = float(value)
            except ValueError:
                pass
            data[current][key] = value
            
    return data

def load_exposure_time(fn: Union[Path, str]) -> float:
    if isinstance(fn, str): fn = Path(fn)
    if fn.is_file():
        pth = fn
    else:
        pth = next(fn.glob('*.ctinfo.xml'))
    lines = pth.read_text().split('\n')
    for line in lines:
        if 'ExposureMilliseconds' in line:
            return float(line.lstrip(' <ExposureMilliseconds>').rstrip('</ExposureMilliseconds>')) / 1000.0
    raise ValueError(f'No exposure time found in {pth}')

def load_from_ang(pth: Path) -> pd.DataFrame:
    txt = pth.read_text()
    lines = txt.split('\n')
    print(f'Loaded {len(lines)} lines from "{pth}"')
    # skip 1st line and load from 2nd with delimiter ':'
    data = {}
    for line in lines[1:]:
        if len(line)<1: continue
        key, val = line.split(':')
        data[int(key)] = float(val)
    df = pd.Series(data).to_frame(name='angles')
    # rename columns and convert to dataframe
    df.index.name = 'indices'
    return df

def gaussian_quadrature_points(n: int) -> Tuple[np.ndarray, np.ndarray]: 
    # returns points and weights for Gaussian quadrature
    # input points are between -1 and 1
    if n == 1:
        return np.array([0]), np.array([1])
    if n == 2:
        return np.array([-1/np.sqrt(3), 1/np.sqrt(3)]), np.array([1, 1])
    if n == 3:
        return np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)]), np.array([5/9, 8/9, 5/9])
    if n == 4:
        return np.array([-np.sqrt(3/7 + 2/7*np.sqrt(6/5)), -np.sqrt(3/7 - 2/7*np.sqrt(6/5)), np.sqrt(3/7 - 2/7*np.sqrt(6/5)), np.sqrt(3/7 + 2/7*np.sqrt(6/5))]), np.array([(18-np.sqrt(30))/36, (18+np.sqrt(30))/36, (18+np.sqrt(30))/36, (18-np.sqrt(30))/36])
    if n == 5:
        return np.array([-1/3*np.sqrt(5 + 2*np.sqrt(10/7)), -1/3*np.sqrt(5 - 2*np.sqrt(10/7)), 0, 1/3*np.sqrt(5 - 2*np.sqrt(10/7)), 1/3*np.sqrt(5 + 2*np.sqrt(10/7))]), np.array([(322 - 13*np.sqrt(70))/900, (322 + 13*np.sqrt(70))/900, 128/225, (322 + 13*np.sqrt(70))/900, (322 - 13*np.sqrt(70))/900])
    raise ValueError(f'Gaussian quadrature points for {n} points not implemented')

def uniform_quadrature_points(n: int) -> Tuple[np.ndarray, np.ndarray]:
    # returns points and weights for uniform quadrature
    # input points are between -1 and 1
    if n == 1:
        return np.array([0]), np.array([1])
    points = np.linspace(-1, 1, n)
    weights = np.ones(n)
    weights[0] = 0.5
    weights[-1] = 0.5
    weights /= (n-1)
    return points, weights

def load_from_ctdata(pth: Path) -> pd.DataFrame:
    txt = pth.read_text()
    lines = txt.split('\n')
    print(f'Loaded {len(lines)} lines from "{pth}"')
    for i, line in enumerate(lines):
        if 'Index' in line:
            break
        if 'Angle(deg)' in line:
            columns = {'Projection': 'indices', 'Angle(deg)': 'angles', 'Time(s)': 'times'}
            break
    df = pd.read_csv(pth, skiprows=i, delim_whitespace=True)
    # rename columns
    df.rename(columns=columns, inplace=True)
    angular_step = np.mean(np.diff(df['angles']))
    df['angles'] = df['angles'] + angle_correction(angular_step)
    # Due to mismatch in formats, loading from _ctdata required negative sign for angle
    df['angles'] = -df['angles']
    df.set_index('indices', inplace=True, drop=True)
    return df

        
def load_angles(fn: Path) -> pd.DataFrame:
    if isinstance(fn, str): fn = Path(fn)
    if fn.is_file():
        pth = fn
        assert pth.exists()
    else:
        files = list(fn.glob('*.ang')) + list(fn.glob('*_ctdata*'))
        assert len(files) == 1
        pth = files[0]
    if 'ang' in pth.suffix:
        return load_from_ang(pth)
    elif 'ctdata' in pth.stem:
        return load_from_ctdata(pth)

def m4(m: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = m
    return out
    
def _pose_to_matrix(theta: float, R: float):
    cam_matrix = np.eye(4)
    
    th_rad = - np.pi * theta / 180 + np.pi/2 # 0 deg when x-axis pointing left
    pos = R * np.array([np.cos(th_rad), np.sin(th_rad), 0])
    phi = np.arctan2(pos[1], pos[0]) + math.radians(90)

    # Blender way
    cam_matrix[:3, 3] = pos
    cam_matrix = cam_matrix@m4(Rotation.from_rotvec(np.pi/2 * np.array([1,0,0])).as_matrix()) # rotate 90 degrees around x
    cam_matrix = cam_matrix@m4(Rotation.from_rotvec(phi * np.array([0,1,0])).as_matrix())
    # Could do the rotations in one go
    # cam_matrix = m4(Rotation.from_euler('XY', [np.pi/2, phi]).as_matrix())@cam_matrix
    # cam_matrix[:3, 3] = pos
    # or
    # cam_matrix = m4(Rotation.from_euler('xz', [np.pi/2, phi]).as_matrix())@cam_matrix
    # cam_matrix = cam_matrix@m4(Rotation.from_euler('XZ', [np.pi/2, phi]).as_matrix())
    # cam_matrix[:3, 3] = pos
    return cam_matrix

def pose_to_matrix(theta: float, R: float):
    """
    Returns a 4x4 camera matrix
    theta: degrees around the +Z axis
    R:radius
    """
    th = - math.radians(theta)
    c, s = math.cos(th), math.sin(th)

    R_wc = np.array([
        [ c,  0,  s],
        [ s,  0, -c],
        [ 0,  1,  0],
    ], dtype=float)
    t_w  = np.array([-R*s, R*c, 0.0], dtype=float)

    cam_matrix = np.eye(4)
    cam_matrix[:3, :3] = R_wc
    cam_matrix[:3, 3]  = t_w
    return cam_matrix

def time_correction(rotation_rate: float):
    tau = np.polyval([-0.0002111, 0.06664], rotation_rate)
    return tau

def angle_correction(angular_step: float):
    # empirical correction
    # angular step in degrees
    return -0.5 * angular_step + 5.0
    
def main(
        folder: Path, 
        images_folder: str = 'images',
        xtekct_file: Optional[str] = None,
        angles_file: Optional[str] = None,
        exposure_file: Optional[str] = None,
        output_fname: Optional[str] = 'transforms.json',
        deblurring: Literal['Gauss', 'uniform', None] = None,
        deblurring_points: int = 7,
        time: Optional[float] = None,
        flat_field: Optional[float] = None,
):

    if xtekct_file is not None:
        data = load_xtekct(folder / xtekct_file)
    else:
        data = load_xtekct(folder)
    H = data['XTekCT']['DetectorPixelsX']*data['XTekCT']['DetectorPixelSizeX'] / 2
    L = data['XTekCT']['SrcToDetector']
    alpha = 2*np.arctan(H/L) #* 180 / np.pi
    scale_factor = 2 / (data['XTekCT']['VoxelSizeX']*data['XTekCT']['VoxelsX'])
    R = data['XTekCT']['SrcToObject'] * scale_factor
    print(f'alpha: {alpha*180/np.pi}, R: {R}, scale_factor: {scale_factor}')

    f = data['XTekCT']['DetectorPixelsX'] / 2 / np.tan(alpha/2)
    out_data = {}
    if flat_field is not None:
        out_data['flat_field'] = flat_field
    out_data.update({
        'camera_angle_x': alpha,
        'w': data['XTekCT']['DetectorPixelsX'],
        'h': data['XTekCT']['DetectorPixelsY'],
        'cx': data['XTekCT']['DetectorPixelsX'] / 2,
        'cy': data['XTekCT']['DetectorPixelsY'] / 2,
        'fl_x': f,
        'fl_y': f,
        'frames': []
    })

    if angles_file is not None:
        angular_data = load_angles(folder/angles_file)
    else:
        angular_data = load_angles(folder)
    
    # angular_data['angles'] = data['XTekCT']['InitialAngle'] + data['XTekCT']['AngularStep'] * np.arange(angular_data.shape[0])
    # angular_data['angles'] = -angular_data['angles']
    

    if deblurring is not None:
        if exposure_file is not None:
            exposure_time = load_exposure_time(folder / exposure_file)
        else:
            exposure_time = load_exposure_time(folder)
        fit = np.polyfit(angular_data['times'], angular_data['angles'], deg=1)
        angular_data['angles_fit'] = np.polyval(fit, angular_data['times'])
        rotation_rate = abs(fit[0])

    min_time = 1<<20
    max_time = -1<<20
    mean_thetas = []

    for fn in sorted((folder/images_folder).glob('*.png')):
        frame_data = {
            'file_path': fn.relative_to(folder).as_posix(),
        }
        
        proj_num = int(fn.stem.split('_')[-1])
        if deblurring is None:
            theta = angular_data.loc[proj_num, 'angles']
            mean_thetas.append(theta)
            cam_matrix = pose_to_matrix(theta, R)
            frame_data['transform_matrix'] = listify_matrix(cam_matrix)
            if time is not None:
                frame_data['time'] = time
            else:
                frame_data['time'] = 0.0
            frame_data['angle'] = theta
        else:
            _time = angular_data.loc[proj_num, 'times']
            if deblurring=='Gauss':
                quad_points, quad_weights = gaussian_quadrature_points(deblurring_points)
            elif deblurring=='uniform':
                quad_points, quad_weights = uniform_quadrature_points(deblurring_points)
            else:
                raise ValueError(f'Unknown deblurring method {deblurring}')
            quad_weights = quad_weights / 2.0 # convert integral to average
            quad_weights = quad_weights.tolist()
            # instead of _time-0.5*exposure_time, start at _time -[DT]*exposure_time (empirical offset)
            tau = time_correction(rotation_rate)
            times = _time - tau - 0.5*exposure_time + (quad_points/2 + 0.5) * exposure_time
            thetas = np.polyval(fit, times)
            mean_thetas.append(np.mean(thetas))
            cam_matrices = [listify_matrix(pose_to_matrix(theta, R)) for theta in thetas]
            if time is not None:
                times = [time] * len(cam_matrices)
            elif 'eval' in fn.stem:
                _time = time if time is not None else 1.0
                times = [_time] * len(cam_matrices)
            else:
                times = times.tolist()
                min_time = min(min_time, min(times))
                max_time = max(max_time, max(times))

            if len(times)==1:
                times = times[0]
                cam_matrices = cam_matrices[0]
                quad_weights = 1.0
            frame_data.update({
                'transform_matrix': cam_matrices,
                'time': times,
                'camera_weights': quad_weights,
                'angle': thetas.tolist(),
            })

        out_data['frames'].append(frame_data)

    print(mean_thetas)

    # normalize time between 0 and 1
    if deblurring is not None and time is None:
        for frame_data in out_data['frames']:
            if 'eval' in frame_data['file_path']:
                continue
            if isinstance(frame_data['time'], list):
                frame_data['time'] = [(t-min_time)/(max_time-min_time) for t in frame_data['time']]
            else:
                frame_data['time'] = (frame_data['time']-min_time)/(max_time-min_time)

    (folder / output_fname).write_text(json.dumps(out_data, indent=2))
    print(f'Saved {(folder / output_fname).as_posix()} with {len(out_data["frames"])} frames')

if __name__ == '__main__':
    tyro.cli(main)
