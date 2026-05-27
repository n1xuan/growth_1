#!/usr/bin/env python3
"""
Simple example of using the Python bindings for xray_projection_render.

Make sure you've built the shared library first:
    cd xray_projection_render
    ./build.sh
"""

import sys
from pathlib import Path

# Add the current directory to the path so we can import xray_renderer
sys.path.insert(0, str(Path(__file__).parent))

from xray_renderer import XRayRenderer

def main():
    # Initialize the renderer
    # It will automatically find the shared library in the build/ directory
    try:
        renderer = XRayRenderer()
        print("✓ Successfully loaded the shared library")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nPlease build the shared library first:")
        print("  cd xray_projection_render")
        print("  ./build.sh")
        sys.exit(1)
    
    # Example 1: Using custom camera angles
    print("\n--- Example 1: Custom Camera Angles ---")
    
    params = {
        'input': 'examples/cube_w_hole.yaml',  # Make sure this file exists
        'output_dir': 'output_images',
        'resolution': 256,  # Lower resolution for faster testing
        'R': 4.0,
        'fov': 40.0,
        'log_level': 'error',  # Quiet operation (only errors). Use 'info' for verbose output
    }
    
    # Define custom camera angles (azimuthal, polar in degrees)
    camera_angles = [
        {'azimuthal': 0, 'polar': 90},    # Looking from positive X axis
        {'azimuthal': 90, 'polar': 90},  # Looking from positive Y axis
        {'azimuthal': 180, 'polar': 90}, # Looking from negative X axis
        {'azimuthal': 270, 'polar': 90}, # Looking from negative Y axis
    ]
    
    print(f"Rendering {len(camera_angles)} views with custom angles...")
    result = renderer.render(params, camera_angles=camera_angles)
    
    if result['success']:
        print(f"✓ Successfully rendered {result['num_images']} images to {result['output_dir']}")
    else:
        print(f"✗ Error: {result.get('error', 'Unknown error')}")
    
    # Example 2: Using equispaced angle generation
    print("\n--- Example 2: Equispaced Angle Generation ---")
    
    params2 = {
        'input': 'examples/cube_w_hole.yaml',
        'output_dir': 'output_images_equispaced',
        'num_images': 4,          # Number of equispaced projections
        'polar_angle': 90.0,      # Fixed polar angle (90 = horizontal plane)
        'resolution': 256,
    }
    
    print(f"Rendering {params2['num_images']} equispaced views...")
    result2 = renderer.render(params2)
    
    if result2['success']:
        print(f"✓ Successfully rendered {result2['num_images']} images to {result2['output_dir']}")
    else:
        print(f"✗ Error: {result2.get('error', 'Unknown error')}")

if __name__ == '__main__':
    main()

