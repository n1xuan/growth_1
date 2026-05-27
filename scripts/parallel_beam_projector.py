"""
Parallel beam X-ray forward projector for synchrotron imaging.

This module replaces the cone beam Go renderer (xray_projection_render) used in
neural_xray for experimental data. In neural_xray's experimental workflow, the
Go renderer generates projections from analytic geometry or XCT-reconstructed
volumes using cone beam geometry (Nikon XTH 225 lab source). For dendrite data
acquired at synchrotron beamlines, the geometry is parallel beam (all rays
parallel), so we need a dedicated projector.

Supports three backends in priority order:
    1. ASTRA Toolbox GPU (parallel3d) — fast, accurate, recommended
    2. scikit-image radon — CPU, slice-by-slice, decent fallback
    3. pure numpy — slow, zero dependencies, last resort

Example usage:
    from parallel_beam_projector import ParallelBeamProjector

    projector = ParallelBeamProjector(volume, voxel_size=1.0)
    projections = projector.project_batch([0.0, 45.0, 90.0, 135.0])
    # projections: (4, Nz, det_width)

    # Verify correctness with FBP round-trip
    recon, psnr = projector.verify_with_fbp(num_angles=180)
"""

import numpy as np
from typing import List, Optional, Tuple


class ParallelBeamProjector:
    """Parallel beam X-ray forward projector for 3D volumes.

    For synchrotron imaging, all rays are parallel. The projection at angle theta
    is a line integral (Radon transform) of the attenuation field along the beam
    direction.

    Convention:
        - Rotation axis: z (vertical)
        - Beam rotates in the x-y plane
        - Detector plane: (u, v) where u = horizontal, v = vertical (z)
        - Output: line integral ∫μ·ds (consistent with neural_xray renderer)
    """

    def __init__(self, volume: np.ndarray, voxel_size: float = 1.0):
        """
        Args:
            volume: 3D numpy array, shape (Nz, Ny, Nx), attenuation field
            voxel_size: physical size of one voxel (um or mm)
        """
        self.volume = volume.astype(np.float32)
        self.Nz, self.Ny, self.Nx = volume.shape
        self.voxel_size = voxel_size
        self._backend = self._detect_backend()

    def _detect_backend(self) -> str:
        """Detect available projection backend."""
        try:
            import astra
            if astra.use_cuda():
                return 'astra_gpu'
            return 'astra_cpu'
        except ImportError:
            pass
        try:
            from skimage.transform import radon
            return 'skimage'
        except ImportError:
            pass
        return 'numpy'

    def project_batch(self, angles_deg: List[float],
                      detector_width: Optional[int] = None) -> np.ndarray:
        """
        Generate parallel beam projections at multiple angles.

        Args:
            angles_deg: rotation angles in degrees
            detector_width: detector width in pixels (None = auto from volume diagonal)

        Returns:
            projections: (num_angles, Nz, det_width), line integral values
        """
        if detector_width is None:
            raw = int(np.ceil(np.sqrt(self.Nx**2 + self.Ny**2))) + 2
            # Round up to nearest multiple of 12 (divisible by 2, 3, 4)
            # This avoids dimension mismatch when dataparser downscales
            detector_width = int(np.ceil(raw / 12) * 12)

        if self._backend.startswith('astra'):
            return self._project_astra(angles_deg, detector_width)
        elif self._backend == 'skimage':
            return self._project_skimage(angles_deg, detector_width)
        else:
            return self._project_numpy(angles_deg, detector_width)

    def _project_astra(self, angles_deg: List[float],
                       detector_width: int) -> np.ndarray:
        """Project using ASTRA Toolbox (GPU-accelerated 3D parallel beam)."""
        import astra

        angles_rad = np.array(angles_deg, dtype=np.float64) * np.pi / 180.0

        vol_geom = astra.create_vol_geom(self.Ny, self.Nx, self.Nz)
        proj_geom = astra.create_proj_geom(
            'parallel3d',
            1.0, 1.0,          # detector pixel spacing (voxel units)
            self.Nz,           # detector rows
            detector_width,    # detector columns
            angles_rad
        )

        proj_id, proj_data = astra.create_sino3d_gpu(
            self.volume, proj_geom, vol_geom
        )
        astra.data3d.delete(proj_id)

        # ASTRA output shape: (Nz, num_angles, det_width) → (num_angles, Nz, det_width)
        proj_data = np.transpose(proj_data, (1, 0, 2))
        proj_data *= self.voxel_size
        return proj_data

    def _project_skimage(self, angles_deg: List[float],
                         detector_width: int) -> np.ndarray:
        """Project using scikit-image radon transform (slice-by-slice CPU)."""
        from skimage.transform import radon

        angles = np.array(angles_deg, dtype=np.float64)
        num_angles = len(angles)
        projections = np.zeros((num_angles, self.Nz, detector_width), dtype=np.float32)

        for iz in range(self.Nz):
            sino = radon(self.volume[iz], theta=angles, circle=False)
            # sino: (det_width_auto, num_angles)
            dw = sino.shape[0]
            if dw >= detector_width:
                offset = (dw - detector_width) // 2
                projections[:, iz, :] = sino[offset:offset+detector_width, :].T
            else:
                offset = (detector_width - dw) // 2
                projections[:, iz, offset:offset+dw] = sino.T

        projections *= self.voxel_size
        return projections

    def _project_numpy(self, angles_deg: List[float],
                       detector_width: int) -> np.ndarray:
        """Pure numpy fallback with bilinear interpolation. Slow but portable."""
        num_angles = len(angles_deg)
        projections = np.zeros((num_angles, self.Nz, detector_width), dtype=np.float32)

        cx, cy = self.Nx / 2.0, self.Ny / 2.0
        cu = detector_width / 2.0

        num_steps = int(np.ceil(np.sqrt(self.Nx**2 + self.Ny**2))) * 2
        s_max = max(self.Nx, self.Ny) * 0.75
        s_vals = np.linspace(-s_max, s_max, num_steps)
        ds = s_vals[1] - s_vals[0]

        for ia, theta_deg in enumerate(angles_deg):
            theta = np.radians(theta_deg)
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            for iu in range(detector_width):
                u = (iu - cu)

                x_vals = u * cos_t + s_vals * sin_t + cx
                y_vals = u * sin_t - s_vals * cos_t + cy

                ix0 = np.floor(x_vals).astype(int)
                iy0 = np.floor(y_vals).astype(int)
                valid = (ix0 >= 0) & (ix0 < self.Nx - 1) & (iy0 >= 0) & (iy0 < self.Ny - 1)

                if not valid.any():
                    continue

                fx = x_vals[valid] - ix0[valid]
                fy = y_vals[valid] - iy0[valid]
                ix_v, iy_v = ix0[valid], iy0[valid]

                for iz in range(self.Nz):
                    vals = (self.volume[iz, iy_v, ix_v] * (1-fx) * (1-fy) +
                            self.volume[iz, iy_v, ix_v+1] * fx * (1-fy) +
                            self.volume[iz, iy_v+1, ix_v] * (1-fx) * fy +
                            self.volume[iz, iy_v+1, ix_v+1] * fx * fy)
                    projections[ia, iz, iu] = np.sum(vals) * ds

        projections *= self.voxel_size
        return projections

    def verify_with_fbp(self, num_angles: int = 180,
                        slice_idx: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """
        Verify projector correctness by forward-project → FBP round-trip.

        Args:
            num_angles: equispaced angles for verification
            slice_idx: which z-slice to test (None = middle)

        Returns:
            (reconstructed_slice, psnr_value)
        """
        if slice_idx is None:
            slice_idx = self.Nz // 2

        angles = np.linspace(0, 180, num_angles, endpoint=False).tolist()
        projs = self.project_batch(angles)
        sino = projs[:, slice_idx, :]  # (num_angles, det_width)

        try:
            from skimage.transform import iradon
            recon = iradon(sino.T, theta=np.array(angles), circle=False)
        except ImportError:
            print('scikit-image not available for FBP verification')
            return np.zeros_like(self.volume[slice_idx]), 0.0

        # Crop to original size
        rh, rw = recon.shape
        oh, ow = self.Ny, self.Nx
        y0 = max(0, (rh - oh) // 2)
        x0 = max(0, (rw - ow) // 2)
        recon = recon[y0:y0+oh, x0:x0+ow]

        original = self.volume[slice_idx, :recon.shape[0], :recon.shape[1]]
        mse = np.mean((original - recon)**2)
        if mse > 0:
            psnr = 10 * np.log10(original.max()**2 / mse)
        else:
            psnr = float('inf')

        return recon, psnr
