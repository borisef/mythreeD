import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    FoVPerspectiveCameras
)
from pytorch3d.structures import Meshes

# Function to render the object
def render_object(camera_params, obj_path, object_location, landmark = None, save_as_img = "out.png"):
    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the 3D model
    mesh = load_objs_as_meshes([obj_path], device=device)

    # Define camera parameters
    R, T = look_at_view_transform(camera_params["dist"], camera_params["elev"], camera_params["azim"])

    R_np = R.cpu().numpy()
    T_np = T.cpu().numpy()

    #cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Define the rasterization settings
    raster_settings = RasterizationSettings(
        image_size=(512,640), #[H,W]
        blur_radius=0.0,
        faces_per_pixel=1,
        max_faces_per_bin = 10,
        bin_size=0
    )

    # Place a point light in front of the object
    lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

    # Define the renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )

    # Apply translation to the mesh
    translation = torch.tensor(object_location, device=device).unsqueeze(0)
   # mesh = mesh.translate(translation) #NOT working

    # Render the image
    images = renderer(mesh)

    # Convert images to numpy and plot
    image = images[0, ..., :3].cpu().numpy()
    plt.imshow(image)
    plt.axis("off")
    plt.show()

    if(save_as_img is not None):
        cv2.imwrite(save_as_img,(image[:,:,::-1]*255))

    return image

# Example usage
camera_params = {
    "dist": 52.7,  # Distance of the camera from the object
    "elev": 20.0,  # Elevation angle in degrees
    "azim": 90.0  # Azimuth angle in degrees
}
obj_path = "data/bixler/bixler.obj"
# camera_params = {
#     "dist": 2.7,  # Distance of the camera from the object
#     "elev": 20.0,  # Elevation angle in degrees
#     "azim": 90.0  # Azimuth angle in degrees
# }
# obj_path = "data/cow_mesh/cow.obj"

object_location = [0.0, 0.0, 0.0]  # [x, y, z] position of the object in the world
landmark = [0.0, 0.0, 0.0]  # [x, y, z] position of the landmark on the object


render_object(camera_params, obj_path, object_location,landmark = landmark, save_as_img = "out1.png")
