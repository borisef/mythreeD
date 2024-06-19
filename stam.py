import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt


def render_and_project(camera_params, obj_path, object_location, landmarks, out_img = 'rendered_image_with_landmarks.png'):
    # Load the 3D model
    mesh = o3d.io.read_triangle_mesh(obj_path)

    # Apply translation to the mesh
    translation = np.array(object_location)
    mesh.translate(translation)

    # Set up the visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(mesh)

    # Camera parameters
    height, width = camera_params["H"], camera_params["W"]
    fov = camera_params["fov"]

    # Calculate the focal length from the FOV
    focal_length = width / (2 * np.tan(np.radians(fov) / 2))

    principal_point = (width / 2, height / 2)
    K = np.array([
        [focal_length, 0, principal_point[0]],
        [0, focal_length, principal_point[1]],
        [0, 0, 1]
    ])

    # Extrinsic parameters (R, T)
    R = o3d.geometry.get_rotation_matrix_from_xyz([
        np.radians(camera_params["elev"]),
        np.radians(camera_params["azim"]),
        0
    ])
    T = np.array([[0, 0, -camera_params["dist"]]]).T
    extrinsic = np.hstack((R, T))
    extrinsic = np.vstack((extrinsic, [0, 0, 0, 1]))

    # Render the scene and get the depth image
    vis.poll_events()
    vis.update_renderer()
    depth = vis.capture_depth_float_buffer(do_render=True)
    depth = np.asarray(depth)

    # Lists to store 2D coordinates and visibility status of landmarks
    landmark_2d_list = []
    visibility_list = []

    for landmark in landmarks:
        # Get the 2D coordinates of the landmark
        landmark_3d = np.array(landmark) + translation
        landmark_3d_h = np.append(landmark_3d, 1)
        landmark_2d_h = K @ (extrinsic @ landmark_3d_h)[:3]
        landmark_2d = landmark_2d_h[:2] / landmark_2d_h[2]

        # Check if the landmark is within the image bounds
        if 0 <= landmark_2d[0] < width and 0 <= landmark_2d[1] < height:
            x, y = int(landmark_2d[0]), int(landmark_2d[1])
            landmark_depth = (extrinsic @ landmark_3d_h)[2]
            visible = np.abs(depth[y, x] - landmark_depth) < 1e-2
        else:
            visible = False

        landmark_2d_list.append(landmark_2d)
        visibility_list.append(visible)

    # Convert depth to color image for visualization
    depth_color = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Plot the image and landmarks
    plt.imshow(depth_color)
    for i, (landmark_2d, visible) in enumerate(zip(landmark_2d_list, visibility_list)):
        color = 'green' if visible else 'red'
        plt.scatter(landmark_2d[0], landmark_2d[1], color=color, s=50)
    plt.axis("off")

    # Save the image
    if(out_img is not None):
        plt.savefig(out_img)

    # Show the image
    plt.show()

    vis.destroy_window()
    return landmark_2d_list, visibility_list


# Example usage
camera_params = {
    "dist": 15.7,  # Distance of the camera from the object
    "elev": 20.0,  # Elevation angle in degrees
    "azim": 10.0,  # Azimuth angle in degrees
    "fov": 45.0,  # Field of view in degrees
    "H": 512,  # Height of the image
    "W": 512  # Width of the image
}
obj_path = "data/cow_mesh/cow.obj"

object_location = [1.0, 0.0, 0.0]  # [x, y, z] position of the object in the world
landmarks = [
    [1.0, 0.0, 0.0],  # First landmark [x, y, z] position
    [0.0, 1.0, 0.0],  # Second landmark [x, y, z] position
    # Add more landmarks as needed
]

landmark_2d_list, visibility_list = render_and_project(camera_params, obj_path, object_location, landmarks, out_img="fov45.png")
for i, (landmark_2d, is_visible) in enumerate(zip(landmark_2d_list, visibility_list)):
    print(f"2D coordinates of landmark {i + 1}:", landmark_2d)
    print(f"Is landmark {i + 1} visible?", is_visible)

