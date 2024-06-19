import cv2
import numpy as np

import cv2
import numpy as np
from scipy.optimize import least_squares
import open3d as o3d
from PIL import Image

# from OpenGL.GL import *
# from OpenGL.GLUT import *
# from OpenGL.GLU import *
from pyrr import Matrix44, Vector3
import pywavefront
import sys


# Function to set up the viewport and perspective
# def init_opengl(width, height):
#     glViewport(0, 0, width, height)
#     glMatrixMode(GL_PROJECTION)
#     glLoadIdentity()
#     gluPerspective(45.0, float(width) / float(height), 0.1, 100.0)
#     glMatrixMode(GL_MODELVIEW)
#     glLoadIdentity()
#     glEnable(GL_DEPTH_TEST)
#
#
# # Function to render the 3D model
# def render_model(obj_path, Rt, K, width, height, output_path):
#     # Initialize GLUT
#     glutInit(sys.argv)
#     glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
#     glutInitWindowSize(width, height)
#     glutInitWindowPosition(100, 100)
#     window = glutCreateWindow("3D Model Renderer")
#
#     # Load the 3D model
#     scene = pywavefront.Wavefront(obj_path)
#
#     # Configure the camera
#     extrinsic = np.linalg.inv(Rt)
#     modelview_matrix = Matrix44(extrinsic.T)
#     projection_matrix = Matrix44(K.T)
#
#     # Set up OpenGL
#     init_opengl(width, height)
#
#     # Render loop
#     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#     glLoadIdentity()
#     glMatrixMode(GL_MODELVIEW)
#
#     # Apply the modelview and projection matrix
#     glLoadMatrixf(modelview_matrix)
#     glMultMatrixf(projection_matrix)
#
#     # Draw the 3D model
#     scene.draw()
#
#     # Read the rendered image from the buffer
#     data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
#     img = Image.frombytes("RGBA", (width, height), data)
#     img = img.transpose(Image.FLIP_TOP_BOTTOM)
#
#     # Save the image
#     img.save(output_path)
#
#     # Clean up
#     glutDestroyWindow(window)


def render_3d_model_to_image(K, Rt, obj_path, image_size, output_path):
    # Load the 3D mesh
    mesh = o3d.io.read_triangle_mesh(obj_path)

    # Create an Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=image_size[1], height=image_size[0], visible=False)

    # Add the mesh to the visualizer
    vis.add_geometry(mesh)

    # Create a camera parameter object
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=image_size[1],
        height=image_size[0],
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2]
    )

    # Convert extrinsic matrix to camera parameters
    extrinsic = np.linalg.inv(Rt)
    camera_params = o3d.camera.PinholeCameraParameters()
    camera_params.intrinsic = intrinsic
    camera_params.extrinsic = extrinsic

    # Set the camera parameters
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(camera_params)

    # Render the image
    vis.poll_events()
    vis.update_renderer()
    img = vis.capture_screen_float_buffer(do_render=True)

    # Convert the image to a format that can be saved
    img_np = np.asarray(img)
    img_np = (img_np * 255).astype(np.uint8)

    # Save the image using PIL
    Image.fromarray(img_np).save(output_path)

    # Close the visualizer
    vis.destroy_window()



def project_points(object_points, rvec, tvec, camera_matrix, dist_coeffs):
    projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    return projected_points.reshape(-1, 2)

def weighted_reprojection_error_func(params, object_points, image_points, camera_matrix, dist_coeffs, weights):
    rvec = params[:3].reshape(3, 1)
    tvec = params[3:].reshape(3, 1)
    projected_points = project_points(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    error = weights * (projected_points - image_points)
    return error.ravel()


def recover_camera_extrinsics_use_weights(object_points, image_points, camera_matrix, dist_coeffs=None, weights=None):
    """
    Recovers the camera extrinsic parameters (rotation matrix and translation vector)
    given object points, image points, camera intrinsic matrix, and optional weights.

    Parameters:
        object_points (np.ndarray): Nx3 array of 3D points in the world space.
        image_points (np.ndarray): Nx2 array of corresponding 2D points in image space.
        camera_matrix (np.ndarray): 3x3 camera intrinsic matrix.
        dist_coeffs (np.ndarray): Distortion coefficients, if any (default is None).
        weights (np.ndarray): Optional Nx1 array of weights for each correspondence (default is None).

    Returns:
        rmat (np.ndarray): 3x3 rotation matrix.
        tvec (np.ndarray): 3x1 translation vector.
        success (bool): Whether the optimization was successful.
        weighted_reprojection_error (float): Weighted reprojection error.
    """

    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    if weights is None:
        weights = np.ones((object_points.shape[0], 1), dtype=np.float32)
    else:
        weights = weights.reshape(-1, 1)

    # Initial guess using solvePnP
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    if not success:
        raise ValueError("PnP solution was not successful.")

    # Convert initial rotation vector and translation to parameter vector
    initial_params = np.hstack((rvec.flatten(), tvec.flatten()))

    # Levenberg-Marquardt optimization
    res = least_squares(
        weighted_reprojection_error_func,
        initial_params,
        args=(object_points, image_points, camera_matrix, dist_coeffs, weights),
        method='lm'
    )

    optimized_params = res.x
    rvec = optimized_params[:3].reshape(3, 1)
    tvec = optimized_params[3:].reshape(3, 1)

    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)

    # Compute final weighted reprojection error
    projected_points = project_points(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    reprojection_error = np.sqrt(np.sum((projected_points - image_points) ** 2, axis=1))
    weighted_reprojection_error = np.sqrt(np.sum((weights * reprojection_error.reshape(-1, 1)) ** 2) / np.sum(weights))

    return rmat, tvec, res.success, weighted_reprojection_error


def recover_camera_extrinsics2(object_points, image_points, camera_matrix, dist_coeffs=None, weights=None):
    """
    Recovers the camera extrinsic parameters (rotation matrix and translation vector)
    given object points, image points, camera intrinsic matrix, and optional weights.

    Parameters:
        object_points (np.ndarray): Nx3 array of 3D points in the world space.
        image_points (np.ndarray): Nx2 array of corresponding 2D points in image space.
        camera_matrix (np.ndarray): 3x3 camera intrinsic matrix.
        dist_coeffs (np.ndarray): Distortion coefficients, if any (default is None).
        weights (np.ndarray): Optional Nx1 array of weights for each correspondence (default is None).

    Returns:
        rmat (np.ndarray): 3x3 rotation matrix.
        tvec (np.ndarray): 3x1 translation vector.
        success (bool): Whether the PnP solution was successful.
        weighted_reprojection_error (float): Weighted reprojection error.
    """

    # Default distortion coefficients if none are provided
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    # Check weights
    if weights is None:
        weights = np.ones((object_points.shape[0], 1), dtype=np.float32)
    else:
        weights = weights.reshape(-1, 1)

    # Use OpenCV's solvePnP to find the extrinsics
   # success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    inliers = []
    aaa = cv2.solvePnPRansac(object_points,image_points,camera_matrix,dist_coeffs,
                                             iterationsCount = 1000,
                                             reprojectionError = 100.0,
                                             confidence = 0.9)
    success=aaa[0];rvec=aaa[1]; tvec=aaa[2]; inliers = aaa[3]
    if not success:
        raise ValueError("PnP solution was not successful.")

    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)

    # Compute weighted reprojection error
    projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected_points = projected_points.reshape(-1, 2)

    # Calculate reprojection error per point
    reprojection_error = np.sqrt(np.sum((projected_points - image_points) ** 2, axis=1))

    # Weighted reprojection error
    weighted_reprojection_error = np.sqrt(np.sum((weights * reprojection_error.reshape(-1, 1)) ** 2) / np.sum(weights))

    return rmat, tvec, success, weighted_reprojection_error, inliers


def convert_fov_to_intrinsics(fov_x,fov_y,W,H):
    # Convert FOV to radians
    fov_x_rad = np.deg2rad(fov_x)
    fov_y_rad = np.deg2rad(fov_y)

    # Calculate focal lengths
    f_x = W / (2 * np.tan(fov_x_rad / 2))
    f_y = H / (2 * np.tan(fov_y_rad / 2))

    # Principal point
    c_x = W / 2
    c_y = H / 2

    # Construct intrinsic matrix
    K = np.array([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1]
    ])

    print("Intrinsic Matrix K:")
    print(K)

    return K


def recover_camera_extrinsics(object_points, image_points, camera_matrix, dist_coeffs = None):
# # Given data
# object_points = np.array([...], dtype=np.float32)  # Nx3 matrix of 3D points
# image_points = np.array([...], dtype=np.float32)  # Nx2 matrix of 2D points
# camera_matrix = np.array([...], dtype=np.float32)  # 3x3 intrinsic matrix
# dist_coeffs = np.zeros((4,1))  # Assuming no lens distortion
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4,1))  # Assuming no lens distortion

    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)

    # rmat is the rotation matrix (3x3)
    # tvec is the translation vector (3x1)
    #TODO: reprojection error
    reproj_error = None
    E = np.concatenate((rmat,tvec),axis=1) # [R|t]

    return (rmat,tvec, success,reproj_error)



if __name__ == "__main__":

    if(0):
        # Given data
        W = 1920  # Image width in pixels
        H = 1080  # Image height in pixels
        fov_x = 60  # Horizontal FOV in degrees
        fov_y = 45  # Vertical FOV in degrees

        K = convert_fov_to_intrinsics(fov_x,fov_y,W,H)
    if(0):
        # Example object points (Nx3) and image points (Nx2)
        object_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [12, 11, 0]], dtype=np.float32)
        image_points = np.array([[100, 100], [200, 100], [100, 200], [200, 200], [300, 300]], dtype=np.float32)

        # Camera intrinsic matrix (example)
        camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)

        # Example weights
        weights = np.array([1.0, 0.8, 1.2, 1.0, 0.01], dtype=np.float32)

        # Recover extrinsic parameters
        rmat, tvec, success, weighted_reprojection_error = recover_camera_extrinsics_use_weights(object_points, image_points,
                                                                                     camera_matrix, weights=weights)

        print("Rotation Matrix:\n", rmat)
        print("Translation Vector:\n", tvec)
        print("PnP Success:", success)
        print("Weighted Reprojection Error:", weighted_reprojection_error)
    if (1):
        # Example object points (Nx3) and image points (Nx2)
        object_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
        image_points = np.array([[100, 100], [200, 100], [100, 200], [200, 200]], dtype=np.float32)

        # Camera intrinsic matrix (example)
        camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)

        # Example weights
        weights = np.array([1.0, 0.8, 1.2, 1.0], dtype=np.float32)

        # Recover extrinsic parameters
        rmat, tvec, success, weighted_reprojection_error = recover_camera_extrinsics2(object_points,
                                                                                                 image_points,
                                                                                                 camera_matrix,
                                                                                                 weights=weights)

        print("Rotation Matrix:\n", rmat)
        print("Translation Vector:\n", tvec)
        print("PnP Success:", success)
        print("Weighted Reprojection Error:", weighted_reprojection_error)

    if(0):
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
            [0, 0, 1, -10],
            [0, 0, 0, 1]
        ])

        # Image size (height, width)
        image_size = (768, 1024)

        # Path to the 3D model
        obj_path = "/home/borisef/projects/pytorch3D/data/cow_mesh/cow.obj"
        #obj_path = "/home/borisef/projects/pytorch3D/data/bixler/bixler.obj"

        # Output path for the PNG image
        output_path = "image2.png"

        #render_3d_model_to_image(K, Rt, obj_path, image_size, output_path)
        render_model(obj_path, Rt, K, 1024, 768, output_path)