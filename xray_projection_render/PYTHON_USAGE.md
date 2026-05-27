# Python Bindings Usage Guide

This guide explains how to use the Python bindings for `xray_projection_render`.

## Prerequisites

1. **Build the shared library**: The Go code must be compiled as a shared library first.
2. **Python 3.6+**: The bindings use standard library modules (`ctypes`, `json`, `pathlib`).

## Step 1: Get the Shared Library

### Option A: Download from Releases (Linux only)

Pre-built Linux (amd64) shared libraries are available in [GitHub releases](https://github.com/igrega348/xray_projection_render/releases). Download `libxray_projection_render_linux-amd64.so` and `libxray_projection_render_linux-amd64.h`, then place them in the `build/` directory.

**Note**: Only Linux shared libraries are available in releases due to CGO cross-compilation limitations. For macOS and Windows, you must build from source (see Option B).

### Option B: Build from Source

If you're on macOS or Windows, or need a different architecture, you'll need to build the shared library from source:

```bash
cd xray_projection_render
./build.sh
```

This will create the shared library in the `build/` directory:
- **macOS**: `build/libxray_projection_render.dylib` (or without extension)
- **Linux**: `build/libxray_projection_render.so`
- **Windows**: `build/xray_projection_render.dll`

**Note**: When building with `-buildmode=c-shared`, Go creates both the library file and a header file (`.h`). The Python bindings only need the library file.

## Step 2: Basic Usage

### Simple Example

```python
import sys
from pathlib import Path

# Add the xray_projection_render directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from xray_renderer import XRayRenderer

# Initialize the renderer
# It will automatically find the shared library in the build/ directory
renderer = XRayRenderer()

# Define parameters
params = {
    'input': 'examples/cube_w_hole.yaml',
    'output_dir': 'output_images',
    'resolution': 512,
    'R': 4.0,
    'fov': 40.0,
}

# Define custom camera angles (azimuthal, polar in degrees)
camera_angles = [
    {'azimuthal': 0, 'polar': 90},
    {'azimuthal': 45, 'polar': 90},
    {'azimuthal': 90, 'polar': 90},
    {'azimuthal': 135, 'polar': 90},
]

# Render the projections
result = renderer.render(params, camera_angles=camera_angles)

if result['success']:
    print(f"Successfully rendered {result['num_images']} images to {result['output_dir']}")
else:
    print(f"Error: {result.get('error', 'Unknown error')}")
```

### Using Equispaced Angle Generation

If you don't provide custom camera angles, the renderer will automatically generate equispaced angles:

```python
from xray_renderer import XRayRenderer

renderer = XRayRenderer()

params = {
    'input': 'examples/cube_w_hole.yaml',
    'output_dir': 'output_images',
    'num_images': 8,          # Number of equispaced projections
    'polar_angle': 90.0,      # Fixed polar angle (90 = horizontal plane)
    'resolution': 512,
}

result = renderer.render(params)
print(result)
```

## Step 3: Specifying Custom Library Path

If the shared library is in a non-standard location, you can specify the path explicitly:

```python
from xray_renderer import XRayRenderer

# Specify the full path to the shared library
renderer = XRayRenderer(library_path='/path/to/libxray_projection_render.dylib')
```

## Complete Parameter Reference

The `render()` method accepts a dictionary with the following parameters:

### Required Parameters
- **`input`** (str): Path to input YAML/JSON file describing the object

### Optional Parameters
- **`output_dir`** (str): Output directory for images (default: `"images"`)
- **`fname_pattern`** (str): Filename pattern with sprintf format (default: `"image_%03d.png"`)
- **`resolution`** (int): Image resolution (default: `512`)
- **`R`** (float): Distance from camera to scene center (default: `4.0`)
- **`fov`** (float): Field of view in degrees (default: `40.0`)
- **`ds`** (float): Integration step size, negative to auto-compute (default: `-1.0`)
- **`density_multiplier`** (float): Density multiplier (default: `1.0`)
- **`flat_field`** (float): Flat field value (default: `0.0`)
- **`integration`** (str): Integration method `"simple"` or `"hierarchical"` (default: `"hierarchical"`)
- **`log_level`** (str): Logging level - `"trace"`, `"debug"`, `"info"`, `"warn"`, `"error"`, `"fatal"`, `"panic"`, or `"disabled"` (default: `"error"` for quiet operation)

### Equispaced Angle Generation Parameters
(Used when `camera_angles` is not provided)
- **`num_images`** (int): Number of images for equispaced angle generation (default: `1`)
- **`out_of_plane`** (bool): Use random polar angles (default: `False`)
- **`polar_angle`** (float): Fixed polar angle in degrees (default: `90.0`)

### Advanced Parameters
- **`jobs_modulo`** (int): Job modulo for parallel execution (default: `1`)
- **`job_num`** (int): Job number for parallel execution (default: `0`)
- **`transforms_file`** (str): Output file for transform parameters (default: `"transforms.json"`)
- **`deformation_file`** (str): Path to deformation file (default: `""`)
- **`time_label`** (float): Time label for metadata (default: `0.0`)
- **`transparency`** (bool): Enable transparency in output (default: `False`)
- **`export_volume`** (bool): Export volume grid (default: `False`)

### Camera Angles
You can provide custom camera angles either:
1. As a separate parameter: `renderer.render(params, camera_angles=[...])`
2. In the params dict: `params['camera_angles'] = [...]`

Each camera angle is a dictionary with:
- **`azimuthal`** (float): Azimuthal angle in degrees (0-360)
- **`polar`** (float): Polar angle in degrees (0-180)

## Example: Generating Views for Different Angles

```python
from xray_renderer import XRayRenderer
import numpy as np

renderer = XRayRenderer()

# Generate 16 views around a circle at different elevations
num_views = 16
camera_angles = []

for i in range(num_views):
    azimuthal = i * 360.0 / num_views
    # Vary polar angle from 60 to 120 degrees
    polar = 60 + 60 * np.sin(i * 2 * np.pi / num_views)
    camera_angles.append({
        'azimuthal': azimuthal,
        'polar': polar
    })

params = {
    'input': 'examples/cube_w_hole.yaml',
    'output_dir': 'spiral_views',
    'resolution': 512,
    'log_level': 'info',  # Set to 'info' to see progress messages, or 'error' for quiet operation
}

result = renderer.render(params, camera_angles=camera_angles)
print(f"Rendered {result['num_images']} views")
```

## Controlling Logging Output

By default, the Python API runs in quiet mode (log level `"error"`), which only shows error messages. This prevents the verbose JSON log messages from cluttering your output.

You can control the logging level using the `log_level` parameter:

```python
# Quiet mode (default) - only errors shown
params = {'input': 'object.yaml', 'log_level': 'error'}

# Show info messages (includes progress and status updates)
params = {'input': 'object.yaml', 'log_level': 'info'}

# Show debug messages (very verbose, includes detailed information)
params = {'input': 'object.yaml', 'log_level': 'debug'}

# Disable all logging
params = {'input': 'object.yaml', 'log_level': 'disabled'}
```

Valid log levels (from least to most verbose):
- `"disabled"` - No logging
- `"error"` - Only errors (default for Python API)
- `"warn"` - Warnings and errors
- `"info"` - Info, warnings, and errors (shows progress messages)
- `"debug"` - Debug, info, warnings, and errors (very verbose)
- `"trace"` - All messages (most verbose)

Note: Log messages are written to stderr, so they won't interfere with the return value from `render()`.

## Troubleshooting

### Library Not Found Error

If you get `FileNotFoundError: Library not found at ...`, check:

1. **Get the shared library**: 
   - **Linux**: Download from [GitHub releases](https://github.com/igrega348/xray_projection_render/releases) (only Linux libraries are available)
   - **macOS/Windows**: Build from source using `./build.sh` or manually (see below)
2. **Check the path**: The bindings look in `build/` directory relative to `xray_renderer.py`
3. **Platform-specific naming**: 
   - macOS: `libxray_projection_render.dylib` or `libxray_projection_render` (no extension)
   - Linux: `libxray_projection_render.so` or `libxray_projection_render_linux-amd64.so` (from releases)
   - Windows: `xray_projection_render.dll`

### Building the Shared Library Manually

If the build script doesn't work, you can build manually:

```bash
# For current platform
go build -buildmode=c-shared -o build/libxray_projection_render .

# For specific platform (example: Linux)
GOOS=linux GOARCH=amd64 go build -buildmode=c-shared -o build/libxray_projection_render .
```

**Note**: On macOS, `-buildmode=c-shared` may create the library without an extension. You may need to rename it or adjust the Python code to find it without the `.dylib` extension.

### Runtime Errors

If rendering fails, check the `result['error']` field for details. Common issues:
- Invalid input file path
- Invalid camera angles (e.g., NaN or out of range)
- Missing dependencies or object files

## Return Value

The `render()` method returns a dictionary with:
- **`success`** (bool): Whether rendering succeeded
- **`error`** (str, optional): Error message if failed
- **`num_images`** (int): Number of images rendered
- **`output_dir`** (str): Output directory path

