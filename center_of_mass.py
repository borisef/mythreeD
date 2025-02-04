import numpy as np
from scipy.ndimage import binary_closing, center_of_mass, distance_transform_edt, binary_opening
from skimage.measure import regionprops, find_contours
import cv2


import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from skimage.measure import find_contours


import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.draw import ellipse

from matplotlib.patches import Ellipse

def fit_ellipse_to_contour(segmentation):
    from skimage.measure import find_contours
    import cv2

    contours = find_contours(segmentation, level=0.5)
    if not contours:
        raise ValueError("No contours found in segmentation.")

    # Select the largest contour (by number of points)
    largest_contour = max(contours, key=len)

    # Convert contour to the format required by OpenCV
    contour_points = np.array([(y, x) for x, y in largest_contour], dtype=np.float32)  # Swap x, y
    contour_points = contour_points.reshape(-1, 1, 2)  # OpenCV requires (N, 1, 2) shape

    if len(contour_points) < 5:
        raise ValueError("Not enough points to fit an ellipse.")

    # Fit ellipse using OpenCV
    ellipse = cv2.fitEllipse(contour_points)

    # Extract ellipse parameters
    (xc, yc), (major_axis, minor_axis), angle = ellipse

    # Fix the orientation
    corrected_angle = 90 - angle  # Convert OpenCV angle to Matplotlib's convention

    return {
        "center": (xc, yc),
        "axes": (major_axis, minor_axis),
        "angle": corrected_angle
    }

from matplotlib.patches import Ellipse

def visualize_with_ellipse_center(segmentation, results, fitted_ellipse):
    closed_segmentation = results["closed_segmentation"]
    bbox = results["bounding_box"]
    ellipse = fitted_ellipse  # Fitted ellipse from OpenCV
    com = results["center_of_mass"]
    max_distance_point = results["max_distance_point"]

    # Extract ellipse parameters
    ellipse_center = ellipse["center"]
    ellipse_axes = (ellipse["axes"][1] / 2, ellipse["axes"][0] / 2)  # Axes (width, height)
    ellipse_angle = ellipse["angle"]

    # Find contours
    contours = find_contours(closed_segmentation, level=0.5)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Original segmentation
    ax[0].imshow(segmentation, cmap="gray")
    ax[0].set_title("Original Segmentation")
    ax[0].axis("off")

    # Processed segmentation
    ax[1].imshow(closed_segmentation, cmap="gray")
    ax[1].set_title("Processed Segmentation")
    ax[1].axis("off")

    # Add bounding box
    min_row, min_col, max_row, max_col = bbox
    rect = Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
                     linewidth=2, edgecolor="red", facecolor="none", label="Bounding Box")
    ax[1].add_patch(rect)

    # Add fitted ellipse
    ell = Ellipse(xy=ellipse_center, width=2 * ellipse_axes[0], height=2 * ellipse_axes[1],
                  angle=ellipse_angle, edgecolor="blue", facecolor="none", linewidth=2, label="Fitted Ellipse")
    ax[1].add_patch(ell)

    # Add ellipse center
    ax[1].scatter(ellipse_center[0], ellipse_center[1], color="purple", label="Ellipse Center", marker="x", s=100, zorder=3)

    # Add center of mass
    ax[1].scatter(com[1], com[0], color="yellow", label="Center of Mass", zorder=3)

    # Add max distance point
    ax[1].scatter(max_distance_point[1], max_distance_point[0], color="green", label="Max Distance Point", zorder=3)

    # Add contours
    for contour in contours:
        ax[1].plot(contour[:, 1], contour[:, 0], color="cyan", linewidth=2, label="Contour")

    # Add legend
    ax[1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def visualize_with_max_enclosed_ellipse(segmentation, results, enclosed_ellipse):
    closed_segmentation = results["closed_segmentation"]
    bbox = results["bounding_box"]
    ellipse = results["ellipse"]
    com = results["center_of_mass"]
    max_distance_point = results["max_distance_point"]

    # Enclosed ellipse parameters
    enclosed_center = enclosed_ellipse["center"]
    enclosed_axes = enclosed_ellipse["axes"]
    enclosed_angle = enclosed_ellipse["angle"]

    # Find contours
    contours = find_contours(closed_segmentation, level=0.5)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Original segmentation
    ax[0].imshow(segmentation, cmap="gray")
    ax[0].set_title("Original Segmentation")
    ax[0].axis("off")

    # Closed segmentation with results
    ax[1].imshow(closed_segmentation, cmap="gray")
    ax[1].set_title("Processed Segmentation")
    ax[1].axis("off")

    # Add bounding box
    min_row, min_col, max_row, max_col = bbox
    rect = Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
                     linewidth=2, edgecolor="red", facecolor="none", label="Bounding Box")
    ax[1].add_patch(rect)

    # Add fitted ellipse
    ellipse_center = ellipse[0]
    ellipse_axes = (ellipse[1][1] / 2, ellipse[1][0] / 2)  # Axes lengths (height, width)
    ellipse_angle = ellipse[2]  # Angle in degrees
    ell = Ellipse(xy=ellipse_center, width=2 * ellipse_axes[0], height=2 * ellipse_axes[1],
                  angle=ellipse_angle, edgecolor="blue", facecolor="none", linewidth=2, label="Fitted Ellipse")
    ax[1].add_patch(ell)

    # Add maximum enclosed ellipse
    max_ell = Ellipse(xy=(enclosed_center[1], enclosed_center[0]),
                      width=2 * enclosed_axes[1], height=2 * enclosed_axes[0],
                      angle=enclosed_angle, edgecolor="magenta", facecolor="none", linewidth=2, label="Max Enclosed Ellipse")
    ax[1].add_patch(max_ell)

    # Add center of mass
    ax[1].scatter(com[1], com[0], color="yellow", label="Center of Mass", zorder=3)

    # Add max distance point
    ax[1].scatter(max_distance_point[1], max_distance_point[0], color="green", label="Max Distance Point", zorder=3)

    # Add contours
    for contour in contours:
        ax[1].plot(contour[:, 1], contour[:, 0], color="cyan", linewidth=2, label="Contour")

    # Add legend
    ax[1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()



def find_maximum_enclosed_ellipse(segmentation):
    # Step 1: Compute distance transform
    distance_map = distance_transform_edt(segmentation)

    # Step 2: Find the pixel with the maximum distance (center of the largest circle)
    max_distance_coords = np.unravel_index(np.argmax(distance_map), distance_map.shape)
    max_distance = distance_map[max_distance_coords]

    # Step 3: Initialize ellipse parameters
    center = max_distance_coords  # Initial ellipse center
    axes = (max_distance, max_distance)  # Start with a circle of radius = max distance
    angle = 0  # Ellipse is axis-aligned initially

    # Step 4: Iteratively adjust the ellipse axes
    while True:
        # Generate points inside the ellipse
        rr, cc = ellipse(center[0], center[1], axes[0], axes[1], shape=segmentation.shape)

        # Check if all points are inside the object
        if np.all(segmentation[rr, cc] == 1):
            break  # Found the largest ellipse that fits
        else:
            # Gradually shrink the axes
            axes = (axes[0] - 1, axes[1] - 1)
            if axes[0] <= 0 or axes[1] <= 0:
                raise ValueError("Could not find a valid enclosed ellipse.")

    return {
        "center": center,
        "axes": axes,
        "angle": angle,
    }


def process_segmentation(segmentation):



    # Step 2: Find bounding box
    props = regionprops(segmentation.astype(int))
    if props:
        bbox = props[0].bbox  # (min_row, min_col, max_row, max_col)
    else:
        raise ValueError("No object found in segmentation.")

    side = int(max(bbox) / 20) * 2 + 1
    side1 = int(max(bbox) / 50) * 2 + 1
    # Step 1: Perform morphological closing to close small holes
    closed_segmentation = binary_closing(segmentation, structure=np.ones((side, side)))
    closed_segmentation = binary_opening(closed_segmentation, structure=np.ones((side, side)))

    # Step 3: Fit an ellipse to the contour
    contours = find_contours(closed_segmentation, level=0.5)
    if contours:
        # Select the largest contour
        largest_contour = max(contours, key=lambda c: len(c))
        # Fit ellipse using OpenCV (convert contour to required format)
        contour_points = np.array(largest_contour, dtype=np.float32)
        if len(contour_points) >= 5:  # Minimum points needed for fitEllipse
            #ellipse = cv2.fitEllipse(contour_points)
            ellipse = fit_ellipse_to_contour(closed_segmentation)

        else:
            raise ValueError("Not enough points to fit an ellipse.")
    else:
        raise ValueError("No contours found in segmentation.")

    # Step 4: Find center of mass
    com = center_of_mass(closed_segmentation)

    # Step 5: Find point with maximum distance from all edges
    distance_map = distance_transform_edt(closed_segmentation)
    max_dist_coords = np.unravel_index(np.argmax(distance_map), distance_map.shape)
    max_distance = distance_map[max_dist_coords]

    return {
        "closed_segmentation": closed_segmentation,
        "bounding_box": bbox,
        "ellipse": ellipse,
        "center_of_mass": com,
        "max_distance_point": max_dist_coords,
        "max_distance": max_distance,
    }

# Example Usage
segmentation = np.zeros((100, 100), dtype=np.uint8)
segmentation[30:70, 30:70] = 1  # Example square object
result = process_segmentation(segmentation)

print("Bounding Box:", result["bounding_box"])
print("Ellipse:", result["ellipse"])
print("Center of Mass:", result["center_of_mass"])
print("Point with Max Distance:", result["max_distance_point"])
print("Max Distance:", result["max_distance"])

if __name__ == "__main__":
    im = cv2.imread("a10.png")
    im.shape
    im1 = (im.mean(axis = 2) <150)*1
    res = process_segmentation(im1)
    ee = find_maximum_enclosed_ellipse(im1)
    #visualize_with_max_enclosed_ellipse(im1,res,ee)
    fitted_ellipse = fit_ellipse_to_contour(res["closed_segmentation"])
    visualize_with_ellipse_center(segmentation, res, fitted_ellipse)

