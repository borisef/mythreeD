import bpy
import math
import numpy as np


def render_3d_model(obj_path, K, Rt, image_size, output_path):
    # Clear existing scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import OBJ file
    bpy.ops.import_scene.obj(filepath=obj_path)

    # Set camera parameters
    bpy.ops.object.camera_add(location=(0, 0, 0))
    camera = bpy.context.object
    camera.data.lens_unit = 'FOV'
    camera.data.angle_x = 2 * math.atan(image_size[1] / (2 * K[0, 0]))  # Horizontal FOV from fx
    camera.rotation_mode = 'QUATERNION'

    # Apply extrinsic parameters to the camera
    loc, quat, scale = camera.matrix_world.decompose()
    loc = Rt[:3, 3]
    quat = Rt[:3, :3].to_quaternion()
    camera.location = loc
    camera.rotation_quaternion = quat

    # Set render resolution
    bpy.context.scene.render.resolution_x = image_size[1]
    bpy.context.scene.render.resolution_y = image_size[0]
    bpy.context.scene.render.resolution_percentage = 100

    # Render scene
    bpy.ops.render.render(write_still=True)

    # Save image
    bpy.data.images['Render Result'].save_render(filepath=output_path)


# Example usage
if __name__ == "__main__":
    # Intrinsic matrix K
    K = np.array([
        [1000, 0, 512],
        [0, 1000, 384],
        [0, 0, 1]
    ])

    # Extrinsic matrix Rt
    Rt = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, -5],
        [0, 0, 0, 1]
    ])

    # Image size (height, width)
    image_size = (768, 1024)

    # Path to the 3D model (make sure to use absolute path or relative to Blender's current directory)
    obj_path = "/home/borisef/projects/pytorch3D/data/cow_mesh/cow.obj"

    # Output path for the PNG image
    output_path = "image.png"

    render_3d_model(obj_path, K, Rt, image_size, output_path)
