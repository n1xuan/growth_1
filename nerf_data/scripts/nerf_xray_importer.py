bl_info = {
    "name": "NeRF-Xray Object Importer",
    "author": "Neural XRay Team",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "File > Import > NeRF-Xray Objects (.json)",
    "description": "Import NeRF-Xray object format JSON files",
    "category": "Import-Export",
}

import bpy
import json
import os
import math
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty
from bpy.types import Operator
import numpy as np

def create_sphere(obj_data):
    center = obj_data.get('center', [0, 0, 0])
    radius = obj_data.get('radius', 1.0)
    rho = obj_data.get('rho', 1.0)
    
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=center)
    obj = bpy.context.active_object
    obj.name = f"Sphere_{len(bpy.data.objects)}"
    
    # Add custom properties
    obj['rho'] = rho
    return obj

def create_cube(obj_data):
    center = obj_data.get('center', [0, 0, 0])
    side = obj_data.get('side', 1.0)
    rho = obj_data.get('rho', 1.0)
    
    bpy.ops.mesh.primitive_cube_add(location=center, scale=[side/2, side/2, side/2])
    obj = bpy.context.active_object
    obj.name = f"Cube_{len(bpy.data.objects)}"
    
    # Add custom properties
    obj['rho'] = rho
    return obj

def create_cylinder(obj_data):
    p0 = obj_data.get('p0', [0, 0, 0])
    p1 = obj_data.get('p1', [0, 0, 1])
    radius = obj_data.get('radius', 1.0)
    rho = obj_data.get('rho', 1.0)
    
    # Calculate height and position
    p0 = np.array(p0)
    p1 = np.array(p1)
    direction = p1 - p0
    height = np.linalg.norm(direction)
    center = (p0 + p1) / 2

    # Create cylinder (by default along Z axis)
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius,
        depth=height,
        location=center
    )
    obj = bpy.context.active_object

    # Calculate rotation to align with direction vector
    # Default cylinder is along Z axis, so we need to rotate from Z to our direction
    z_axis = np.array([0, 0, 1])
    direction_normalized = direction / height
    
    # Calculate rotation axis and angle using cross product and dot product
    rotation_axis = np.cross(z_axis, direction_normalized)
    if np.allclose(rotation_axis, 0):
        # If vectors are parallel, we either don't need rotation or need 180° around X axis
        if direction_normalized[2] < 0:
            obj.rotation_euler = (math.pi, 0, 0)
    else:
        # Calculate rotation angle and axis for non-parallel case
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        cos_angle = np.dot(z_axis, direction_normalized)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        # Convert axis-angle to euler rotation
        from mathutils import Matrix, Vector
        rotation_matrix = Matrix.Rotation(angle, 4, Vector(rotation_axis))
        obj.rotation_euler = rotation_matrix.to_euler()

    obj.name = f"Cylinder_{len(bpy.data.objects)}"
    
    # Add custom properties
    obj['rho'] = rho
    obj['p0'] = p0.tolist()
    obj['p1'] = p1.tolist()
    return obj

def create_box(obj_data):
    center = obj_data.get('center', [0, 0, 0])
    sides = obj_data.get('sides', [1, 1, 1])
    rho = obj_data.get('rho', 1.0)
    
    bpy.ops.mesh.primitive_cube_add(location=center)
    obj = bpy.context.active_object
    obj.scale = [s/2 for s in sides]
    obj.name = f"Box_{len(bpy.data.objects)}"
    
    # Add custom properties
    obj['rho'] = rho
    return obj

def create_parallelepiped(obj_data):
    origin = obj_data.get('origin', [0, 0, 0])
    v1 = np.array(obj_data.get('v1', [1, 0, 0]))
    v2 = np.array(obj_data.get('v2', [0, 1, 0]))
    v3 = np.array(obj_data.get('v3', [0, 0, 1]))
    rho = obj_data.get('rho', 1.0)
    
    # Create unit cube
    bpy.ops.mesh.primitive_cube_add(location=[0, 0, 0])
    obj = bpy.context.active_object
    
    # Create transformation matrix
    transform_matrix = np.column_stack([v1, v2, v3, origin])
    transform_matrix = np.vstack([transform_matrix, [0, 0, 0, 1]])
    
    # Apply transformation
    obj.matrix_world = transform_matrix.T
    obj.name = f"Parallelepiped_{len(bpy.data.objects)}"
    
    # Add custom properties
    obj['rho'] = rho
    obj['origin'] = origin
    obj['v1'] = v1.tolist()
    obj['v2'] = v2.tolist()
    obj['v3'] = v3.tolist()
    return obj

def create_object(obj_data):
    obj_type = obj_data.get('type', '').lower()
    
    if obj_type == 'sphere':
        return create_sphere(obj_data)
    elif obj_type == 'cube':
        return create_cube(obj_data)
    elif obj_type == 'cylinder':
        return create_cylinder(obj_data)
    elif obj_type == 'box':
        return create_box(obj_data)
    elif obj_type == 'parallelepiped':
        return create_parallelepiped(obj_data)
    elif obj_type == 'object_collection':
        # Recursively handle collections
        for sub_obj in obj_data.get('objects', []):
            create_object(sub_obj)
    else:
        print(f"Warning: Unsupported object type: {obj_type}")
    return None

class ImportNeRFXrayObjects(Operator, ImportHelper):
    """Import NeRF-Xray Objects from JSON"""
    bl_idname = "import_scene.nerf_xray_objects"
    bl_label = "Import NeRF-Xray Objects"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".json"
    filter_glob: StringProperty(
        default="*.json",
        options={'HIDDEN'},
        maxlen=255,
    )

    def execute(self, context):
        with open(self.filepath, 'r') as f:
            data = json.load(f)
            
        # Handle both single object and collection formats
        if isinstance(data, list):
            objects_data = data
        else:
            objects_data = [data]

        for obj_data in objects_data:
            create_object(obj_data)

        return {'FINISHED'}

def menu_func_import(self, context):
    self.layout.operator(ImportNeRFXrayObjects.bl_idname, text="NeRF-Xray Objects (.json)")

def register():
    bpy.utils.register_class(ImportNeRFXrayObjects)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)

def unregister():
    bpy.utils.unregister_class(ImportNeRFXrayObjects)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)

if __name__ == "__main__":
    register() 