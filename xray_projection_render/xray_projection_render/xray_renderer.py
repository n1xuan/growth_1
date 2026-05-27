"""
Python bindings for xray_projection_render using ctypes.

This module provides a Python interface to the xray_projection_render Go library.
The library must be built as a shared library first using build.sh.

Example usage:
    from xray_renderer import XRayRenderer
    
    # Initialize the renderer
    renderer = XRayRenderer()
    
    # Set up parameters
    params = {
        'input': 'examples/cube_w_hole.yaml',
        'output_dir': 'images',
        'resolution': 512,
        'camera_angles': [
            {'azimuthal': 0, 'polar': 90},
            {'azimuthal': 45, 'polar': 90},
            {'azimuthal': 90, 'polar': 90},
        ],
        'R': 4.0,
        'fov': 40.0,
    }
    
    # Render
    result = renderer.render(params)
    print(result)
"""

import ctypes
import json
import os
import platform
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Try to import resources for finding installed package files
try:
    from importlib import resources
    HAS_IMPORTLIB_RESOURCES = True
except ImportError:
    HAS_IMPORTLIB_RESOURCES = False

try:
    import pkg_resources
    HAS_PKG_RESOURCES = True
except ImportError:
    HAS_PKG_RESOURCES = False

# For downloading from GitHub releases
try:
    import urllib.request
    import urllib.error
    import tempfile
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False


class XRayRenderer:
    """Python wrapper for the xray_projection_render Go library."""
    
    def __init__(self, library_path: Optional[str] = None):
        """
        Initialize the renderer by loading the shared library.
        
        Args:
            library_path: Path to the shared library file. If None, attempts to
                         find it in common locations relative to this file.
        """
        download_error = None
        if library_path is None:
            library_path, download_error = self._find_library()
        
        if not os.path.exists(library_path):
            error_msg = f"Library not found at {library_path}."
            if download_error:
                error_msg += f"\n\nDownload from GitHub releases failed: {download_error}"
            error_msg += "\n\nPlease either:"
            error_msg += "\n1. Build the shared library using build.sh, or"
            error_msg += "\n2. Ensure you have an internet connection to download it from GitHub releases."
            raise FileNotFoundError(error_msg)
        
        self.lib = ctypes.CDLL(library_path)
        self._setup_function_signatures()
    
    def _find_library(self) -> Tuple[str, Optional[str]]:
        """Find the shared library file based on the current platform.
        
        Returns:
            Tuple[str, Optional[str]]: (library_path, download_error_message)
        """
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        if system == "darwin":
            # On macOS, Go may create the library without .dylib extension
            lib_names = ["libxray_projection_render.dylib", "libxray_projection_render"]
        elif system == "windows":
            lib_names = ["xray_projection_render.dll"]
        else:  # Linux and others
            # On Linux, Go may create the library without .so extension
            lib_names = ["libxray_projection_render.so", "libxray_projection_render"]
        
        # First, try to find library in installed package location
        installed_path = self._find_installed_library(lib_names)
        if installed_path:
            return installed_path, None
        
        # Fallback to development mode: look relative to this file
        script_dir = Path(__file__).parent.resolve()
        build_dir = script_dir / "build"
        
        # Try each possible library name in package build directory
        for lib_name in lib_names:
            lib_path = build_dir / lib_name
            if lib_path.exists():
                return str(lib_path), None
            
            # Fallback: try current directory
            lib_path = script_dir / lib_name
            if lib_path.exists():
                return str(lib_path), None
        
        # Also check parent directory's build folder (project root)
        parent_build_dir = script_dir.parent / "build"
        for lib_name in lib_names:
            lib_path = parent_build_dir / lib_name
            if lib_path.exists():
                return str(lib_path), None
        
        # If not found locally, try downloading from GitHub releases
        downloaded_path, download_error = self._download_from_github_release(lib_names)
        if downloaded_path:
            return downloaded_path, None
        
        # Return expected path for error message
        return str(build_dir / lib_names[0]), download_error
    
    def _find_installed_library(self, lib_names: List[str]) -> Optional[str]:
        """Try to find the library in the installed package location."""
        try:
            # First try pkg_resources (works with setuptools)
            if HAS_PKG_RESOURCES:
                try:
                    dist = pkg_resources.get_distribution("xray-renderer")
                    # Get the package installation directory
                    # For installed packages, location points to site-packages
                    package_dir = Path(dist.location) / "xray_projection_render"
                    build_dir = package_dir / "build"
                    
                    for lib_name in lib_names:
                        lib_path = build_dir / lib_name
                        if lib_path.exists():
                            return str(lib_path)
                except (pkg_resources.DistributionNotFound, AttributeError, ImportError):
                    # Package not installed via pip
                    pass
            
            # Try using importlib.resources (Python 3.9+)
            if HAS_IMPORTLIB_RESOURCES:
                try:
                    # Use files() API for Python 3.9+
                    if hasattr(resources, 'files'):
                        package = resources.files("xray_projection_render")
                        build_dir = package / "build"
                        
                        for lib_name in lib_names:
                            try:
                                lib_path = build_dir / lib_name
                                # Convert Traversable to Path for existence check
                                lib_path_str = str(lib_path)
                                if Path(lib_path_str).exists():
                                    return lib_path_str
                            except (AttributeError, TypeError, Exception):
                                continue
                except (ModuleNotFoundError, TypeError, AttributeError):
                    # Not installed as a package, or resources API not available
                    pass
        except Exception:
            # Any other error, fall back to development mode
            pass
        
        return None
    
    def _download_from_github_release(self, lib_names: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """Try to download the library from GitHub releases if not found locally.
        
        Returns:
            Tuple[Optional[str], Optional[str]]: (library_path, error_message)
        """
        if not HAS_URLLIB:
            return None, "urllib not available"
        
        try:
            system = platform.system().lower()
            machine = platform.machine().lower()
            
            # Map platform to GitHub release asset name
            # Based on build.sh, assets are named like: libxray_projection_render_<platform>-<arch>
            if system == "darwin":
                if machine in ["arm64", "aarch64"]:
                    asset_name = "libxray_projection_render_darwin-arm64"
                else:
                    asset_name = "libxray_projection_render_darwin-amd64"
            elif system == "windows":
                asset_name = "libxray_projection_render_windows-amd64.dll"
            elif system == "linux":
                asset_name = "libxray_projection_render_linux-amd64.so"
            else:
                # Unknown platform, skip download
                return None, f"Unknown platform: {system}/{machine}"
            
            # Try to get version from package metadata or git
            version = self._get_package_version()
            if not version:
                # Fallback to latest release
                version = "latest"
            else:
                # Check if this is a development/pre-release version
                # Development versions (e.g., 1.5.0.dev6, 1.5.0a1, 1.5.0rc1) don't have GitHub releases
                # Use "latest" instead for these versions
                version_lower = version.lower()
                if any(indicator in version_lower for indicator in ['.dev', 'dev', 'a', 'alpha', 'b', 'beta', 'rc', 'pre']):
                    version = "latest"
            
            # Determine cache directory
            cache_dir = self._get_cache_directory()
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine expected library name for this platform
            if system == "darwin":
                expected_name = "libxray_projection_render.dylib"
            elif system == "windows":
                expected_name = "xray_projection_render.dll"
            else:  # Linux
                expected_name = "libxray_projection_render.so"
            
            cached_lib = cache_dir / expected_name
            
            # Check if already cached
            if cached_lib.exists():
                return str(cached_lib), None
            
            # Download from GitHub releases
            if version == "latest":
                url = f"https://github.com/igrega348/xray_projection_render/releases/latest/download/{asset_name}"
            else:
                # Remove 'v' prefix if present
                version_tag = version.lstrip('v')
                url = f"https://github.com/igrega348/xray_projection_render/releases/download/v{version_tag}/{asset_name}"
            
            print(f"Downloading library from GitHub releases: {asset_name}")
            try:
                urllib.request.urlretrieve(url, str(cached_lib))
                # Make executable on Unix systems
                if system != "windows":
                    os.chmod(cached_lib, 0o755)
                print(f"✓ Downloaded library to {cached_lib}")
                return str(cached_lib), None
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    error_msg = f"Asset '{asset_name}' not found at {url}"
                    if version != "latest":
                        error_msg += f" (tried version {version})"
                    return None, error_msg
                error_msg = f"HTTP error {e.code}: {e.reason}"
                if cached_lib.exists():
                    cached_lib.unlink()  # Clean up partial download
                return None, error_msg
            except urllib.error.URLError as e:
                error_msg = f"Network error: {str(e)}"
                if cached_lib.exists():
                    cached_lib.unlink()  # Clean up partial download
                return None, error_msg
            except Exception as e:
                error_msg = f"Download failed: {str(e)}"
                if cached_lib.exists():
                    cached_lib.unlink()  # Clean up partial download
                return None, error_msg
                
        except Exception as e:
            # Any other error
            return None, f"Unexpected error during download: {str(e)}"
    
    def _get_package_version(self) -> Optional[str]:
        """Get the package version from installed package or git tag."""
        # Try to get version from installed package
        if HAS_PKG_RESOURCES:
            try:
                dist = pkg_resources.get_distribution("xray-renderer")
                return dist.version
            except (pkg_resources.DistributionNotFound, AttributeError):
                pass
        
        # Try to get version from _version.py (setuptools_scm)
        try:
            script_dir = Path(__file__).parent.resolve()
            version_file = script_dir / "_version.py"
            if version_file.exists():
                # Read version from _version.py
                with open(version_file, 'r') as f:
                    for line in f:
                        if line.startswith('version = '):
                            version = line.split('=')[1].strip().strip('"').strip("'")
                            return version
        except Exception:
            pass
        
        # Try to get from git tag
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'describe', '--tags', '--always'],
                cwd=Path(__file__).parent.parent,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                tag = result.stdout.strip()
                # Extract version from tag (e.g., "v1.4-2-g5fc76f6" -> "v1.4")
                if tag.startswith('v'):
                    version = tag.split('-')[0]
                    return version
        except Exception:
            pass
        
        return None
    
    def _get_cache_directory(self) -> Path:
        """Get the cache directory for downloaded libraries."""
        system = platform.system().lower()
        # Use user cache directory
        if system == "windows":
            cache_base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
        elif system == "darwin":
            cache_base = Path.home() / 'Library' / 'Caches'
        else:  # Linux and others
            cache_base = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
        
        return cache_base / 'xray-renderer' / 'libs'
    
    def _setup_function_signatures(self):
        """Set up the function signatures for ctypes."""
        # RenderProjections: takes a C string, returns a C string
        self.lib.RenderProjections.argtypes = [ctypes.c_char_p]
        self.lib.RenderProjections.restype = ctypes.POINTER(ctypes.c_char)
        
        # FreeString: frees a C string
        self.lib.FreeString.argtypes = [ctypes.c_char_p]
        self.lib.FreeString.restype = None
    
    def render(
        self,
        params: Dict,
        camera_angles: Optional[List[Dict[str, float]]] = None
    ) -> Dict:
        """
        Render X-ray projections based on the provided parameters.
        
        Args:
            params: Dictionary containing render parameters. Supported keys:
                - input: Path to input YAML/JSON file describing the object (required)
                - output_dir: Output directory for images (default: "images")
                - fname_pattern: Filename pattern with sprintf format (default: "image_%03d.png")
                - resolution: Image resolution (default: 512)
                - num_images: Number of images for equispaced angle generation (default: 1)
                - out_of_plane: Use random polar angles (default: False)
                - ds: Integration step size, negative to auto-compute (default: -1.0)
                - R: Distance from camera to scene center (default: 4.0)
                - fov: Field of view in degrees (default: 40.0)
                - jobs_modulo: Job modulo for parallel execution (default: 1)
                - job_num: Job number for parallel execution (default: 0)
                - transforms_file: Output file for transform parameters (default: "transforms.json")
                - deformation_file: Path to deformation file (default: "")
                - time_label: Time label for metadata (default: 0.0)
                - transparency: Enable transparency in output (default: False)
                - export_volume: Export volume grid (default: False)
                - polar_angle: Fixed polar angle in degrees (default: 90.0)
                - density_multiplier: Density multiplier (default: 1.0)
                - flat_field: Flat field value (default: 0.0)
                - integration: Integration method "simple" or "hierarchical" (default: "hierarchical")
                - log_level: Logging level - "trace", "debug", "info", "warn", "error", "fatal", "panic", or "disabled" (default: "error" for quiet operation)
            camera_angles: Optional list of camera angle dictionaries with 'azimuthal' and 'polar' keys.
                          If provided, overrides num_images/out_of_plane/polar_angle parameters.
        
        Returns:
            Dictionary with render results:
                - success: Boolean indicating success
                - error: Error message if failed
                - num_images: Number of images rendered
                - output_dir: Output directory path
        """
        # Prepare parameters dict
        render_params = {
            "input": params.get("input"),
            "output_dir": params.get("output_dir", "images"),
            "fname_pattern": params.get("fname_pattern", "image_%03d.png"),
            "resolution": params.get("resolution", 512),
            "num_images": params.get("num_images", 1),
            "out_of_plane": params.get("out_of_plane", False),
            "ds": params.get("ds", -1.0),
            "R": params.get("R", 4.0),
            "fov": params.get("fov", 40.0),
            "jobs_modulo": params.get("jobs_modulo", 1),
            "job_num": params.get("job_num", 0),
            "transforms_file": params.get("transforms_file", "transforms.json"),
            "deformation_file": params.get("deformation_file", ""),
            "time_label": params.get("time_label", 0.0),
            "transparency": params.get("transparency", False),
            "export_volume": params.get("export_volume", False),
            "polar_angle": params.get("polar_angle", 90.0),
            "density_multiplier": params.get("density_multiplier", 1.0),
            "flat_field": params.get("flat_field", 0.0),
            "integration": params.get("integration", "hierarchical"),
            "log_level": params.get("log_level", "error"),  # Default to quiet (only errors)
            "camera_angles": [],
        }
        
        # Handle camera_angles parameter
        if camera_angles is not None:
            render_params["camera_angles"] = camera_angles
        elif "camera_angles" in params:
            render_params["camera_angles"] = params["camera_angles"]
        
        # Validate required parameters
        if not render_params["input"]:
            raise ValueError("'input' parameter is required")
        
        # Convert to JSON string
        params_json = json.dumps(render_params)
        params_bytes = params_json.encode('utf-8')
        
        # Call the C function
        result_ptr = self.lib.RenderProjections(params_bytes)
        
        # Convert result back to Python string
        result_str = ctypes.string_at(result_ptr).decode('utf-8')
        
        # Free the C string
        self.lib.FreeString(result_ptr)
        
        # Parse and return result
        return json.loads(result_str)

