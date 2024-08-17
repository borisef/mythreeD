import cv2
import numpy as np
from PIL import Image, ImageDraw
import trimesh

import aux

def project_points(points, K, Rt):
    # Apply extrinsic parameters (Rt) to points
    points = np.dot(Rt[:3, :3], points) + Rt[:3, 3].reshape(3, 1)

    # Apply intrinsic parameters (K) to project points onto image plane
    points_proj = np.dot(K, points)

    # Normalize projected points
    points_proj = points_proj[:2, :] / points_proj[2, :]

    return points_proj


def render_3d_model(obj_path, K, Rt, image_size, output_path):
    # Load the 3D mesh model
    mesh = trimesh.load(obj_path)

    # Get vertices of the mesh
    vertices = np.array(mesh.vertices.T)  # Transpose to have shape (3, num_vertices)

    # Project vertices onto the image plane
    vertices_proj = project_points(vertices, K, Rt)

    # Create image
    image = Image.new('RGB', (image_size[1], image_size[0]), color='white')
    draw = ImageDraw.Draw(image)

    # Draw the mesh onto the image
    for face in mesh.faces:
        points = [tuple(vertices_proj[:, vertex]) for vertex in face]
        draw.polygon(points, outline='black', fill='white')

    # Save the image
    image.save(output_path)


# Example usage
if __name__ == "__main__":
    # Intrinsic matrix K
    K = np.array([
        [1000, 0, 512],
        [0, 1000, 384],
        [0, 0, 1]
    ], dtype = np.float32)

    # Extrinsic matrix Rt
    Rt = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, -5],
        [0, 0, 0, 1]
    ])

    # Image size (height, width)
    image_size = (768, 1024)

    # Path to the 3D model (obj_path)
   #obj_path = "/home/borisef/projects/pytorch3D/data/teapot/teapot.obj"
   # obj_path = "/home/borisef/projects/pytorch3D/data/bixler/bixler.obj"
    obj_path = "/home/borisef/projects/pytorch3D/data/cow_mesh/cow.obj"

    # Output path for the PNG image
    output_path = "data/output/image.png"

    render_3d_model(obj_path, K, Rt, image_size, output_path)

    kpts3D = {'Nose':[0, 0.376,-0.648],
              'Tail':[0, -0.09, 1.04],
              'Leg_front_left':[-0.23, -0.733,0.026],
              'Leg_front_right':[ 0.23, -0.718, -0.039],
              'Ear_right': [0.46, 0.73, -0.23],
              'Ear_left': [-0.46, 0.73, -0.23]
              }
    # kpts2D = [[100,120],
    #           [50,70],
    #           [90,22],
    #           [11,100]]

    object_points = np.array([kpts3D["Nose"],kpts3D["Tail"], kpts3D["Ear_right"], kpts3D["Ear_left"],
                              kpts3D["Leg_front_left"], kpts3D["Leg_front_right"]
                              ], dtype=np.float32)

    #image_points = np.array([[512, 322], [507, 404], [421, 247], [600, 245],  [557, 556], [465, 557]], dtype=np.float32)
    #image_points = image_points[:,::-1]

    image_points = np.array([[200, 222], [507, 404], [421, 247], [100, 245], [557, 556], [465, 557]], dtype=np.float32)
    #image_points = np.array([[300, 200], [300, 300], [400, 400], [350, 255], [300, 350], [350, 300]], dtype=np.float32)

    # Camera intrinsic matrix (example)
    camera_matrix = K #np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)

    # Example weights
    weights = np.array([1.0, 0.8, 1.2, 1.0, 1, 1], dtype=np.float32)

    # Recover extrinsic parameters
    rmat, tvec, success,stam, inliers = aux.recover_camera_extrinsics2(object_points,
                                                            image_points,
                                                            camera_matrix,
                                                            weights = weights)

    print("Rotation Matrix:\n", rmat)
    print("Translation Vector:\n", tvec)
    print("PnP Success:", success)
   # print("Weighted Reprojection Error:", weighted_reprojection_error)

    #newRt = Rt
    temp_vec = np.array([[0, 0, 0, 1]], dtype=np.float32)
    temp = np.concatenate([rmat, tvec], axis=1)
    newRt = np.concatenate([temp,temp_vec],axis=0)

    output_path1 = "data/output/image1.png"

    render_3d_model(obj_path, K, newRt, image_size, output_path1)

    img = cv2.imread(output_path1)

    for i in range(image_points.shape[0]):
        thecolor = (0,255,10)
        x = int(image_points[i,0])
        y = int(image_points[i,1])
        if( i not in inliers):
            thecolor = (0,0,255)

        cv2.drawMarker(img, (x, y), thecolor, markerType=cv2.MARKER_CROSS,
                       markerSize=10, thickness=2)

    cv2.imwrite(output_path1, img)



