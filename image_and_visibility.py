import cv2
import numpy as np
import trimesh
import pyrender
import os
from PIL import Image, ImageDraw, ImageFont
import math


os.environ["PYOPENGL_PLATFORM"] = "egl"

mycolors = [
    [255,0,0],
    [0,255,0],
    [0,0,255],
    [200,200,10],
    [200,0,250]
]*10


def compute_largest_vertex_distance(mesh_file):
    """
    Computes the largest distance between any two vertices in a 3D mesh.

    Parameters:
    - mesh_file: Path to the mesh file (e.g., .obj, .ply).

    Returns:
    - max_distance: The largest distance between any two vertices.
    """
    # Load the mesh
    mesh = trimesh.load(mesh_file)

    # Get the vertices as a NumPy array
    vertices = mesh.vertices

    # Compute the pairwise distances using broadcasting for efficiency
    diff = vertices[:, np.newaxis, :] - vertices[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)

    # Find the maximum distance
    max_distance = np.max(distances)

    return max_distance

def compute_object_bounding_box(obj_file, camera_matrix, rvec, tvec, image_size):
    """
    Compute the 2D bounding box of the entire object in the image plane.

    Parameters:
    - mesh: The 3D mesh object (Trimesh object).
    - camera_matrix: Intrinsic camera matrix (3x3).
    - rvec, tvec: Extrinsic camera parameters (rotation and translation vectors).
    - image_size: Size of the output image (width, height).

    Returns:
    - bbox: (x_min, y_min, x_max, y_max) bounding box in pixel coordinates.
    """
    # Get the vertices of the mesh
    mesh = trimesh.load(obj_file)
    vertices = mesh.vertices

    # Project all vertices into the image plane
    vertices_homogeneous = vertices.reshape(-1, 3)
    projected_points, _ = cv2.projectPoints(vertices_homogeneous, rvec, tvec, camera_matrix, distCoeffs=None)
    projected_points = projected_points.squeeze()

    # Ensure the points are within image bounds
    x_coords = np.clip(projected_points[:, 0], 0, image_size[0] - 1)
    y_coords = np.clip(projected_points[:, 1], 0, image_size[1] - 1)

    # Compute the bounding box
    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

    return (x_min, y_min, x_max, y_max)


def draw_keypoints_extended1(input_image_path, output_image_path, keypoints, draw_keypoint_name=False):
    # Open the input image
    img = Image.open(input_image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Define marker styles
    marker_radius = 5
    label_offset = 20  # Base distance between label and marker
    grid_size = 20  # Size of grid cell for collision detection

    # Load a default font for drawing text
    try:
        font = ImageFont.truetype("arial.ttf", 12)  # Use system font
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font

    # Create a grid to track occupied areas
    width, height = img.size
    grid = [[False for _ in range((height // grid_size) + 1)] for _ in range((width // grid_size) + 1)]

    def is_position_free(label_x, label_y):
        """Check if the grid cell at the label position is free."""
        grid_x = int(label_x // grid_size)
        grid_y = int(label_y // grid_size)
        return not grid[grid_x][grid_y]

    def mark_grid_occupied(label_x, label_y):
        """Mark the grid cell at the label position as occupied."""
        grid_x = int(label_x // grid_size)
        grid_y = int(label_y // grid_size)
        grid[grid_x][grid_y] = True

    def find_non_occluding_position(x, y):
        """Find a non-overlapping position for the label."""
        for radius in range(label_offset, 200, grid_size):  # Incremental radius
            for angle in range(0, 360, 30):  # Check positions at various angles
                label_x = x + radius * math.cos(math.radians(angle))
                label_y = y + radius * math.sin(math.radians(angle))
                if (0 <= label_x < width and 0 <= label_y < height) and is_position_free(label_x, label_y):
                    mark_grid_occupied(label_x, label_y)
                    return label_x, label_y
        # If no free position, return a default far position
        return x + label_offset, y + label_offset

    for key, value in keypoints.items():
        coordinates = value[1]
        visibility = value[2]
        color = tuple(value[3])  # Convert color list to tuple
        x, y = coordinates

        # Draw the marker
        if visibility == 1:
            # Draw a visible circle
            draw.ellipse(
                [(x - marker_radius, y - marker_radius), (x + marker_radius, y + marker_radius)],
                outline=color, fill=color
            )
        else:
            # Draw an invisible "X" marker
            offset = marker_radius
            draw.line([(x - offset, y - offset), (x + offset, y + offset)], fill=color, width=2)
            draw.line([(x - offset, y + offset), (x + offset, y - offset)], fill=color, width=2)

        # Draw the keypoint name if the flag is set
        if draw_keypoint_name:
            label_x, label_y = find_non_occluding_position(x, y)
            draw.text((label_x, label_y), key, fill=color, font=font)
            # Draw an arrow connecting the label to the marker
            draw.line([(x, y), (label_x, label_y)], fill=color, width=1)

    # Save the new image
    img.save(output_image_path)
    print(f"Markers drawn and saved to {output_image_path}")

def draw_keypoints_extended(input_image_path, output_image_path, keypoints, draw_keypoint_name=False):
    # Open the input image
    img = Image.open(input_image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Define marker styles
    marker_radius = 5
    label_offset = 20  # Distance between label and marker

    # Load a default font for drawing text
    try:
        font = ImageFont.truetype("arial.ttf", 12)  # Use system font
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font

    used_positions = []  # To keep track of used label positions

    def find_non_occluding_position(x, y):
        """Find a non-overlapping position for the label."""
        for angle in range(0, 360, 30):  # Check positions at various angles
            label_x = x + label_offset * math.cos(math.radians(angle))
            label_y = y + label_offset * math.sin(math.radians(angle))
            # Check for overlap with previous positions
            overlap = any(
                math.hypot(label_x - ux, label_y - uy) < label_offset / 2 for ux, uy in used_positions
            )
            if not overlap:
                used_positions.append((label_x, label_y))
                return label_x, label_y
        # If all else fails, return original offset position
        return x + label_offset, y

    for key, value in keypoints.items():
        coordinates = value[1]
        visibility = value[2]
        color = tuple(value[3])  # Convert color list to tuple
        x, y = coordinates

        # Draw the marker
        if visibility == 1:
            # Draw a visible circle
            draw.ellipse(
                [(x - marker_radius, y - marker_radius), (x + marker_radius, y + marker_radius)],
                outline=color, fill=color
            )
        else:
            # Draw an invisible "X" marker
            offset = marker_radius
            draw.line([(x - offset, y - offset), (x + offset, y + offset)], fill=color, width=2)
            draw.line([(x - offset, y + offset), (x + offset, y - offset)], fill=color, width=2)

        # Draw the keypoint name if the flag is set
        if draw_keypoint_name:
            label_x, label_y = find_non_occluding_position(x, y)
            draw.text((label_x, label_y), key, fill=color, font=font)
            # Draw an arrow connecting the label to the marker
            draw.line([(x, y), (label_x, label_y)], fill=color, width=1)

    # Save the new image
    img.save(output_image_path)
    print(f"Markers drawn and saved to {output_image_path}")


def draw_keypoints(input_image_path, output_image_path, keypoints, draw_keypoint_name=True):
    # Open the input image
    img = Image.open(input_image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Define marker styles
    marker_radius = 5

    # Load a default font for drawing text
    try:
        font = ImageFont.truetype("arial.ttf", 12)  # Use system font
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font

    for key, value in keypoints.items():
        coordinates = value[1]
        visibility = value[2]
        color = tuple(value[3])  # Convert color list to tuple
        x, y = coordinates

        if visibility == 1:
            # Draw a visible circle
            draw.ellipse(
                [(x - marker_radius, y - marker_radius), (x + marker_radius, y + marker_radius)],
                outline=color, fill=color
            )
        else:
            # Draw an invisible "X" marker
            offset = marker_radius
            draw.line([(x - offset, y - offset), (x + offset, y + offset)], fill=color, width=2)
            draw.line([(x - offset, y + offset), (x + offset, y - offset)], fill=color, width=2)

        # Draw the keypoint name if the flag is set
        if draw_keypoint_name:
            text_position = (x + marker_radius + 2, y - marker_radius - 2)
            draw.text(text_position, key, fill=color, font=font)

    # Save the new image
    img.save(output_image_path)
    print(f"Markers drawn and saved to {output_image_path}")


def render_mesh_with_keypoints(obj_file, camera_matrix, rvec, tvec, image_size, keypoints_dict, kpt_radius):
    """
    Renders the mesh with the given keypoints and returns the rendered image.

    Parameters:
    - obj_file: Path to the 3D object file (mesh).
    - camera_matrix: Intrinsic camera matrix (3x3).
    - rvec, tvec: Extrinsic camera parameters (rotation and translation vectors).
    - image_size: Size of the output image (width, height).
    - keypoints_dict: Dictionary of keypoints (key = name, value = [x, y, z]).
    - kpt_radius: Radius of the keypoints (represented as spheres).

    Returns:
    - color_image: The rendered image (with keypoints as red spheres on the mesh).
    """

    # Load the mesh
    mesh = trimesh.load(obj_file)

    # Create a simple material for the mesh (Gray color)
    mesh_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.0, 0.0, 0.0, 1.0],  # Gray color for the object mesh
        metallicFactor=0.0,
        roughnessFactor=1.0
    )
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, material=mesh_material)

    # Create a scene
    scene = pyrender.Scene()

    # Add the mesh to the scene
    scene.add(mesh_pyrender)

    # For each keypoint, create a plain red sphere and add it to the scene
    for name, position in keypoints_dict.items():
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=kpt_radius)
        sphere.apply_translation(position)

        # Create a red material with ambient lighting
        sphere_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.0, 0.0, 1.0, 1.0],  # Plain red color
            metallicFactor=0.0,
            roughnessFactor=1.0,  # Smooth, but ambient
            emissiveFactor= 0.0,
            smooth=True

            #ambientOcclusionFactor=0.3  # Add some ambient light for visibility
        )
        sphere_mesh = pyrender.Mesh.from_trimesh(sphere, material=sphere_material)
        scene.add(sphere_mesh)

    # Define camera parameters
    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]
    camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)

    # Set camera pose from rotation and translation vectors
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = rotation_matrix
    camera_pose[:3, 3] = tvec

    # Add the camera to the scene
    scene.add(camera, pose=camera_pose)

    # Set up lighting (minimal lighting for visibility)
    light = pyrender.PointLight(color=np.ones(3), intensity=100.0)
    light_pose = np.eye(4)
    light_pose[:3, 3] = [0, 0, 5]
    scene.add(light, pose=light_pose)



    # Render the scene
    renderer = pyrender.OffscreenRenderer(image_size[0], image_size[1])
    color, depth = renderer.render(scene)

    return color, depth

def render_mesh_with_keypoints_old(obj_file, camera_matrix, rvec, tvec, image_size, keypoints_dict, kpt_radius):
    """
    Renders the mesh with the given keypoints and returns the rendered image.

    Parameters:
    - obj_file: Path to the 3D object file (mesh).
    - camera_matrix: Intrinsic camera matrix (3x3).
    - rvec, tvec: Extrinsic camera parameters (rotation and translation vectors).
    - image_size: Size of the output image (width, height).
    - keypoints_dict: Dictionary of keypoints (key = name, value = [x, y, z]).
    - kpt_radius: Radius of the keypoints (represented as spheres).

    Returns:
    - color_image: The rendered image (with keypoints as red spheres on the mesh).
    """

    # Load the mesh
    mesh = trimesh.load(obj_file)

    # Create material for mesh (Gray color)
    mesh_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.5, 0.5, 0.5, 1.0],  # Gray color
        metallicFactor=0.5,
        roughnessFactor=0.8
    )
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, material=mesh_material)

    # Create a scene
    scene = pyrender.Scene()

    # Add the mesh to the scene
    scene.add(mesh_pyrender)

    # For each keypoint, create a red sphere and add it to the scene
    for name, position in keypoints_dict.items():
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=kpt_radius)
        sphere.apply_translation(position)

        # Red color for keypoints
        sphere_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.0, 0.0, 1.0, 1.0],  # Red color
            metallicFactor=0.0,
            roughnessFactor=1.0
        )
        sphere_mesh = pyrender.Mesh.from_trimesh(sphere, material=sphere_material)
        scene.add(sphere_mesh)

    # Define camera parameters
    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]
    camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)

    # Set camera pose from rotation and translation vectors
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = rotation_matrix
    camera_pose[:3, 3] = tvec

    # Add the camera to the scene
    scene.add(camera, pose=camera_pose)

    # Set up lighting
    light = pyrender.PointLight(color=np.ones(3), intensity=100.0)
    light_pose = np.eye(4)
    light_pose[:3, 3] = [0, 0, 5]
    scene.add(light, pose=light_pose)

    # Render the scene
    renderer = pyrender.OffscreenRenderer(image_size[0], image_size[1])
    color, depth = renderer.render(scene)

    return color, depth

def check_for_red_pixels(image):
    """Check if any pixel in the image is close to red vs black or white."""
    to_red = np.sum(np.power(np.abs(image[:, :, :3] -[0, 0, 255]),2),axis = 2)
    to_white = np.sum(np.power(np.abs(image[:, :, :3] -[255, 255, 255]),2),axis = 2)
    to_black = np.sum(np.power(np.abs(image[:, :, :3] - [0, 0, 0]),2),axis = 2)

    A = np.array((to_red<to_white) & (to_red<to_black), dtype = np.uint8)
    if(A.sum()>3):
        return 1
    else:
        return 0


def compute_visibility(obj_file, camera_matrix, rvec, tvec, image_size, one_keypoint, kpt_radius, kpt_color,
                       output_image):
    """
    Renders the mesh with a single keypoint as a red sphere, and checks if the keypoint is visible.

    Parameters:
    - obj_file: path to the 3D object file (mesh).
    - camera_matrix: intrinsic camera matrix (3x3).
    - rvec, tvec: extrinsic camera parameters (rotation and translation vectors).
    - image_size: size of the output image (width, height).
    - one_keypoint: the 3D keypoint position as [x, y, z].
    - kpt_radius: radius of the keypoint sphere.
    - kpt_color: color of the keypoint (should be close to red).
    - output_image: path where to save the rendered image for debugging.

    Returns:
    - visibility: 1 if the keypoint is visible, 0 if occluded.
    """
    # Render the mesh with this single keypoint (red sphere)
    color_image, _ = render_mesh_with_keypoints(obj_file, camera_matrix, rvec, tvec, image_size,
                                                {'keypoint': one_keypoint}, kpt_radius)

    # Save the debug image
    cv2.imwrite(output_image, color_image)

    # Check if any red pixels are in the rendered image
    if check_for_red_pixels(color_image):
        return 1  # The keypoint is visible
    else:
        return 0  # The keypoint is occluded



def render_mesh_with_keypoints_with_shading(obj_file, camera_matrix, rvec, tvec, image_size, kpts3D,
                                            kpt_radius, kpt_color, draw_kpts = False):
    # Load the mesh
    mesh = trimesh.load(obj_file)

    # Enhance material properties for the mesh
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],  # Bright white
        metallicFactor=0.5,
        roughnessFactor=1.0
    )
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, material=material)

    # Create a scene
    scene = pyrender.Scene()

    # Add the mesh to the scene
    scene.add(mesh_pyrender)

    # Compute camera pose
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = rotation_matrix
    camera_pose[:3, 3] = tvec

    # Compute visibility
    # visibility = compute_visibility(mesh, camera_pose, kpts3D, kpt_radius)

    if(draw_kpts):
        # Add keypoints as spheres
        for idx, (name, position) in enumerate(kpts3D.items()):
            sphere = trimesh.creation.icosphere(subdivisions=3, radius=kpt_radius)
            sphere.apply_translation(position)

            # Set sphere color: visible (red) or occluded (blue)
            color = kpt_color
            sphere_material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=color + [1.0],  # Add alpha
                metallicFactor=0.0,
                roughnessFactor=0.8
            )
            sphere_mesh = pyrender.Mesh.from_trimesh(sphere, material=sphere_material)
            scene.add(sphere_mesh)

    # Define camera parameters
    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]
    camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)

    # Add the camera to the scene
    scene.add(camera, pose=camera_pose)

    # Add lights to the scene
    main_light = pyrender.PointLight(color=np.ones(3), intensity=100.0)
    main_light_pose = np.eye(4)
    main_light_pose[:3, 3] = [0, 0, 5]
    scene.add(main_light, pose=main_light_pose)

    # Render the scene
    renderer = pyrender.OffscreenRenderer(image_size[0], image_size[1])
    color, depth = renderer.render(scene)

    return color


def load_mesh(file_path):
    # Load 3D mesh using trimesh
    mesh = trimesh.load(file_path)
    vertices = np.array(mesh.vertices)  # Extract vertices
    return vertices


def project_points(vertices, camera_matrix, rvec, tvec):
    # Project 3D points to 2D using camera parameters
    projected_points, _ = cv2.projectPoints(vertices, np.array(rvec,dtype=np.float32), np.array(tvec,dtype = np.float32), camera_matrix, None)
    return projected_points.reshape(-1, 2)


def render_image(projected_points, image_size):
    # Create a blank image
    image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

    # Draw points on the image
    for point in projected_points:
        cv2.circle(image, (int(point[0]), int(point[1])), radius=2, color=(255, 255, 255), thickness=-1)

    return image


# Parameters
width = height = 400
obj_file = "/home/borisef/projects/pytorch3D/data/cow_mesh/cow1.obj"
#camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # Intrinsic matrix
camera_matrix = np.array([[1145.915590, 0, 200], [0, 1145.915590, 200], [0, 0, 1]], dtype=np.float32)
rvec = np.array([0, 0, 0.1], dtype = np.float32)  # Rotation vector
tvec = np.array([0, 0, 15], dtype = np.float32)  # Translation vector
image_size = (width, height)  # Image width and height

kpts3D = {  'Nose':[0, 0.376,-0.648],
            'Tail':[0, -0.09, 1.04],
            'Leg_front_left':[-0.23, -0.733,0.026],
            'Leg_front_right':[ 0.23, -0.718, -0.039],
            'Ear_right': [0.46, 0.73, -0.23],
            'Ear_left': [-0.46, 0.73, -0.23],
            'stam':[0.2,0.2,-1],
            'stam1': [0.2, 0.21, -1]
              }
kpt_color = [0,0,1]
kpt_radius = 0.1



bb = compute_object_bounding_box(obj_file, camera_matrix, rvec, tvec, image_size)
#(x_min, y_min, x_max, y_max)
L = max(bb[2] - bb[0],bb[3] - bb[1])
print(bb)
print(L)


dd = compute_largest_vertex_distance(obj_file)
kpt_radius = dd/20 #automatic radius

# Process
vertices = load_mesh(obj_file)
projected_points = project_points(vertices, camera_matrix, rvec, tvec)
rendered_image = render_image(projected_points, image_size)

# Display or save the result
# cv2.imshow("Rendered Image", rendered_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite("/home/borisef/projects/pytorch3D/data/output/rendered_image_v.png", rendered_image)

output_im_path = "/home/borisef/projects/pytorch3D/data/output/rendered_image_v1.png"
# Render the mesh
if(1):
    rendered_image = render_mesh_with_keypoints_with_shading(obj_file, camera_matrix, rvec, tvec, image_size, kpts3D= kpts3D, kpt_radius=kpt_radius,kpt_color = kpt_color)
    cv2.imwrite(output_im_path, rendered_image)


for i,k in enumerate(kpts3D):

    visibility = compute_visibility(
        obj_file=obj_file,
        camera_matrix=camera_matrix,
        rvec=rvec,  # No rotation
        tvec=tvec,  # Camera at (0, 0, 5)
        image_size=image_size,
        one_keypoint=kpts3D[k],
        kpt_radius=kpt_radius, #TODO: compute auto 0.1 of size of object
        kpt_color=[0, 0, 1],  # Red color
        output_image="/home/borisef/projects/pytorch3D/data/output/debug_image.png"
    )
    projected_point = project_points(np.array(kpts3D[k]), camera_matrix, rvec, tvec)

    print("Visibility:", visibility)
    projected_point[0][0] = image_size[0] - projected_point[0][0]
    projected_point[0][1] = image_size[1] - projected_point[0][1]
    kpts3D[k] = [kpts3D[k], projected_point[0].tolist(),visibility, mycolors[i]]

print(kpts3D)
output_im_path_with_markers = "/home/borisef/projects/pytorch3D/data/output/rendered_image_markers.png"
#draw_keypoints(output_im_path,output_im_path_with_markers,kpts3D,draw_keypoint_name= True)
draw_keypoints_extended(output_im_path,output_im_path_with_markers,kpts3D,draw_keypoint_name= True)
#draw_keypoints_extended1(output_im_path,output_im_path_with_markers,kpts3D,draw_keypoint_name= True)
