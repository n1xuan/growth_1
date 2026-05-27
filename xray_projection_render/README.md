# X-ray projection engine

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/igrega348/xray_projection_render/HEAD?urlpath=%2Ftree%2Fexamples%2Fdemo.ipynb)

Go renderer for generating X-ray projections of simple objects.

The projections are saved as images and corresponding `transforms.json` file is produced which can be used in NeRF framework.

The framework the compositions of simple objects (e.g. sphere, cylinder, box, cube, parallelepiped) as well as periodic tessellations.
The gif below (left) is a stack of projections for an object created as a composition of a cube $(\rho=1)$, a sphere $(\rho=-1)$, and a cylinder $(\rho=-1)$. 
On the right is a stack of projections of the same object with a deformation field _(examples/deformation_linear.yaml)_ applied. [1]

![out_f](https://github.com/user-attachments/assets/21c0b6d2-65ee-4762-b615-9ad053aef82e)

## Installation

### Pre-built Executables

Pre-built executables for macOS (Apple Silicon), Windows (x64), and Linux (x64) are available in [GitHub releases](https://github.com/igrega348/xray_projection_render/releases). Simply download the appropriate executable for your platform.

### Building from Source

To build from source, you'll need Go 1.22.1 or later:

```bash
git clone https://github.com/igrega348/xray_projection_render.git
cd xray_projection_render
go build -o xray_projection_render main.go api.go
```

### Python Bindings

For Python usage, you'll need the shared library:
- **Linux users**: Pre-built shared libraries are available in [GitHub releases](https://github.com/igrega348/xray_projection_render/releases)
- **macOS/Windows users**: You must build the shared library from source (see [Python Usage Guide](PYTHON_USAGE.md))

**Note**: Only Linux shared libraries are included in releases due to CGO cross-compilation limitations. macOS and Windows users need to build natively on their platforms.

## Quickstart

The renderer can be used either from the command line (CLI) or programmatically from Python.

### Command Line Interface (CLI)

A single projection at the default resolution (512x512) can be generated using either of the following:
```
# running source code
go run . --input examples/cube_w_hole.yaml
# executable on Windows
xray_projection_render.exe --input examples/cube_w_hole.yaml
# executable on Linux
./xray_projection_render --input examples/cube_w_hole.yaml
```

To see all available options, run:
```
./xray_projection_render -h
```

### Python Interface

The Python interface requires a shared library. Pre-built Linux shared libraries are available in [GitHub releases](https://github.com/igrega348/xray_projection_render/releases). For other platforms, you'll need to build the shared library from source (see [Python Usage Guide](PYTHON_USAGE.md)).

**Note**: Only Linux (amd64) shared libraries are available in releases due to CGO cross-compilation limitations. For macOS and Windows, you'll need to build the shared library natively on those platforms.

```python
from xray_renderer import XRayRenderer

renderer = XRayRenderer()

# Render with custom camera angles
params = {
    'input': 'examples/cube_w_hole.yaml',
    'output_dir': 'images',
    'resolution': 512,
}
camera_angles = [
    {'azimuthal': 0, 'polar': 90},
    {'azimuthal': 45, 'polar': 90},
]

result = renderer.render(params, camera_angles=camera_angles)
```

For detailed Python usage instructions, see [PYTHON_USAGE.md](PYTHON_USAGE.md).

## Highlighted features

### Accepted input formats and object library

Both standard `json` and `yaml` files are accepted as input descriptors of objects. See examples provided in the `examples` folder.
The supported objects have the following attributes

| Object type                         | Attibutes |
|--------------------------------|-----------|
| sphere         | center[3], radius, rho, type=sphere |
| cylinder      | p0[3], p1[3], radius, rho, type=cylinder |
| cube          | center[3], side, rho, type=cube |
| box           | center[3], sides[3], rho, type=box |
| parallelepiped | origin[3], v0[3], v1[3], v2[3], rho, type=parallelepiped |
| gyroid        | center[3], scale, thickness, rho, type=gyroid |
| object_collection | objects[*], type=object_collection |
| unit_cell     | objects::object_collection, xmin, xmax, ymin, ymax, zmin, zmax, type=unit_cell |
| tessellated_obj_coll | uc::unit_cell, xmin, xmax, ymin, ymax, zmin, zmax, type=tessellated_obj_coll |

Where appropriate, attribute types are denoted by `::`, and corresponding array length by square brackets `[]`.
All object types can be found in the examples provided in the `examples` directory.

**Note on the gyroid object**: The gyroid is a triply periodic minimal surface defined by the equation sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) < C. The `scale` parameter controls the periodicity, while `thickness=C` controls the thickness of the surface. The gyroid can be used with `tessellated_obj_coll` to limit the x,y,z max and min.

### Simulate tomographic X-ray projections

Given an object file (e.g. `examples/cube_w_hole.yaml`), X-ray projections can be generated using either the CLI or Python interface.

**Command Line:**
```
go run . --input examples/cube_w_hole.yaml --num_projections 16
```

**Python:**
```python
from xray_renderer import XRayRenderer

renderer = XRayRenderer()
params = {'input': 'examples/cube_w_hole.yaml', 'num_images': 16}
result = renderer.render(params)
```

This will generate 16 equispaced projections in the horizontal plane, as well as `transforms.json` file describing the locations of the corresponding cameras. 
This file is readable using standard NeRF packages.

**Camera Angle Control:**

The renderer supports two modes for specifying camera angles:

1. **Equispaced angle generation** (automatic): Specify the number of projections and optionally a fixed polar angle. The azimuthal angles are automatically spaced evenly around the circle.
   - CLI: Use `--num_projections` and optionally `--polar_angle`
   - Python: Use `num_images` parameter in the params dictionary

2. **Custom angles**: Manually specify the exact azimuthal and polar angle for each projection.
   - CLI: Use `--azimuthal_angles` and `--polar_angles` flags (comma-separated lists)
   - Python: Use `camera_angles` parameter with a list of angle dictionaries

The polar angle (elevation angle) can be controlled in several ways:
- Default: Fixed at 90° (horizontal plane)
- Random angles: Use `--out_of_plane` (CLI) or `out_of_plane: True` (Python) to generate projections at random elevation angles
- Custom angle: Use `--polar_angle` (CLI) or `polar_angle` parameter (Python) to set a specific elevation angle in degrees
- Per-projection: Specify exact angles in the `camera_angles` list (Python) or angle lists (CLI)

While random angles are not typical for X-ray computed tomography, they can be useful as a test set for the evaluation of NeRF reconstruction.

### Hierarchical integration (ray tracing)

The attenuation equation for X-ray comes from the _Beer-Lambert Law_:

$$
I = I_0 \exp\left(-\int_{s_0}^{s_1} \rho(s) \mathrm{ds} \right)
$$

where $I_0$ is the intensity of the empty image and $I$ is the attenuated intensity recorded at the detector/camera.
The ray is assumed to go between $s_0$ and $s_1$.

Two functions are implemented for integration along the ray: _integrate_hierarchical_ (default) and _integrate_along_ray_.
Both are controlled by parameter `--ds`.

_integrate_along_ray_ is simple numerical integration where the ray is divided into equal segments of length $ds$ and the integral is approximated as the sum $\int \rho(s) \mathrm{ds} \approx \sum \rho(s) ds$

_integrate_hierarchical_ is a little bit more advanced. The motivation is to avoid banding artifacts without uniformly reducing $ds$. It works by only refining $ds$ in segments in which the density changed between the left and right boundary.

Normally, `ds` is calculated automatically from the dimensions on the objects. Its value is displayed to the output if `-v` (verbose) flag is active and it may need to be reduced if banding artifacts are observed.

### Deformations

It is possible to apply topology-preserving deformations to the object by using `--deformation_file` with an appropriate deformation descriptor file (yaml).
For example, one can run

```
go run . --input examples/cube_w_hole.yaml --deformation_file examples/deformation_sigmoid.yaml
```

When querying density field, coordinates will be remapped using the chosen deformation field.
Implemented deformations are _Sigmoid_, _Rigid_, _Linear_, _Gaussian_ with the following deformation fields:
- Rigid: parametrized by three-component constant displacement vector $[u_x,u_y,u_z]$.
  
$$
x\leftarrow x+u_x; \quad y\leftarrow y+u_y; \quad z\leftarrow z+u_z
$$

- Linear: parametrized by 6 strains $[\epsilon_{xx}, \epsilon_{yy}, \epsilon_{zz}, \epsilon_{yz}, \epsilon_{xz}, \epsilon_{xy}]$.

$$
\begin{bmatrix} x \\\ y \\\ z \end{bmatrix} \leftarrow \begin{bmatrix} x \\\ y \\\ z \end{bmatrix} + \begin{bmatrix} \epsilon_{xx} & \epsilon_{xy} & \epsilon_{xz} \\\ \epsilon_{xy} & \epsilon_{yy} & \epsilon_{yz} \\\ \epsilon_{xz} & \epsilon_{yz} & \epsilon_{zz} \end{bmatrix} \begin{bmatrix} x \\\ y \\\ z \end{bmatrix}
$$

- Sigmoid: parametrized by _amplitude A, center c, lengthscale L, direction_. For direction 'z':

$$
x \leftarrow x; \quad y \leftarrow y; \quad z \leftarrow z+\frac{A}{1+\exp\left( - (z-c)/L \right) }
$$

- Gaussian: parametrized by _amplitudes A[3], sigmas s[3], centers c[3]_.

$$
r \leftarrow \sqrt{(x-c[0])^2+(y-c[1])^2+(z-c[2])^2}
$$

$$
x \leftarrow x+A[0]\exp\left(-\frac{r^2}{2s[0]^2}\right); \quad y \leftarrow y+A[1]\exp\left(-\frac{r^2}{2s[1]^2}\right); \quad z \leftarrow z+A[2]\exp\left(-\frac{r^2}{2s[2]^2}\right)
$$

### Splitting of long jobs

While projections of simple objects are generated within seconds, for some complex volumes (e.g. fully resolved lattices) at high resolutions, it can take several minutes to generate each projection.
For these reasons, the program can be run in parallel, with each instance only rendering projections based on modulo arithmetic.

For example, suppose we wish to generate 256 projections as 8 parallel jobs. This can be done by running 8 commands independently

```
go run . --input object.yaml --jobs_modulo 8 --job [x] --transforms_file transforms_[x].json
```
where `[x]` goes from 0 to 7.
The images will be saved independently, and all the transforms file can be combined in a post-processing step.

Option `--text_progress` can be set for purely text-based indication of render progress for each image, instead of the deafult progress bar for the whole job; can be useful when running on servers.

### Exporting volume grids

We add a simple voxel exporter. If flag `--export_volume` is set, the executable will produce a voxel grid `volume.raw` at the end of rendering.
The output is a simple binary array of length `res*res*res` and type `UINT8`, with dimensions arranged in `ZXY` order.
If only voxel grid is required as output, one can set `--num_projections 0`



## Command line options

| Option                    | Explanation                                                                                           |
|---------------------------|-------------------------------------------------------------------------------------------------------|
| --output_dir [str]        | Output directory to save the images (default: "images")                                               |
| --input [str]             | Input yaml file describing the object                                                                 |
| --num_projections [int]   | Number of projections to generate (default: 1)                                                        |
| --resolution [int]        | Resolution of the square output images (default: 512)                                                 |
| --out_of_plane            | Generate out of plane projections (random polar angle)                                                |
| --polar_angle [float]     | Set custom polar angle in degrees (cannot be used with out_of_plane flag) (default: 90.0)            |
| --azimuthal_angles [str]  | Comma-separated list of azimuthal angles in degrees (e.g., "0,45,90,135"). Must be used with --polar_angles |
| --polar_angles [str]      | Comma-separated list of polar angles in degrees (e.g., "90,90,90,90"). Must be used with --azimuthal_angles |
| --fname_pattern [str]     | Sprintf pattern for output file name (default: "image_%03d.png")                                      |
| --ds [float]                | Integration step size. If negative, try to infer from smallest feature size in the input file (default: -1) |
| -R [float]                  | Distance between camera and centre of scene (default: 4)                                              |
| --fov [float]               | Field of view in degrees (default: 40)                                                                |
| --integration [str]       | Integration method to use. Options are 'simple' or 'hierarchical'. (default: "hierarchical")          |
| --flat_field [float]        | Flat field value to add to all pixels (default: 0)                                                    |
| --jobs_modulo [int]       | Number of jobs which are being run independently (e.g. jobs_modulo=4 will render every 4th projection) (default: 1) |
| --job [int]               | Job number to run (e.g. job=1 with jobs_modulo=4 will render projections 1, 5, 9, ...) (default: 0)   |
| --transforms_file [str]   | Output file to save the transform parameters (default: "transforms.json")                             |
| --density_multiplier [float] | Multiply all densities by this number (default: 1)                                                    |
| --deformation_file [str]  | File containing deformation parameters                                                                |
| --time_label [float]        | Label to pass to image metadata (default: 0)                                                          |
| --text_progress           | Use text progress bar                                                                                 |
| --transparency            | Enable transparency in output images                                                                  |
| --export_volume            | Export voxel grid of resolution `res x res x res` from density. Save into file `volume.raw`          |
| -v                        | Enable verbose logging                                                                                |
| --help, -h                | Show help                                                                                             |

**Note**: All these parameters are also available in the Python interface. Pass them as keys in the parameters dictionary to `renderer.render()`. See [PYTHON_USAGE.md](PYTHON_USAGE.md) for detailed Python API documentation and examples.

## Footnotes

[1] The projections for the gif were generated by running:
```
go run . --input examples/cube_w_hole.yaml --num_projections 64 --ds 0.01 --fname_pattern 'cube_%03d.png'
go run . --input examples/cube_w_hole.yaml --num_projections 64 --ds 0.01 --fname_pattern 'cube_def_%03d.png' --deformation_file examples/deformation_linear.yaml
```
