import copy

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import time
SLEEP_PAUSE= 0.1


def gen_stabilities(min_stability,max_stability,delta_stability, num_frames, num_points):
    last_stability = np.random.rand(num_points)*(max_stability - min_stability) + min_stability
    all_stabilities = []
    all_stabilities.append(copy.deepcopy((last_stability)))
    for i in range(num_frames):
        last_stability = last_stability + np.random.rand(num_points)*(2*delta_stability) - delta_stability
        last_stability = last_stability.clip(0,1)
        all_stabilities.append(copy.deepcopy((last_stability)))

    return all_stabilities


def project_points(
    points_3d, FOV_H, FOV_W, H, W, t_vec, r_vec, small_std_xy, large_std_xy, p, stabilities
):
    """
    Projects 3D points to a 2D camera plane with optional errors.

    Args:
        points_3d (np.ndarray): 3D points of size (K, 3).
        FOV_H (float): Horizontal field of view in degrees.
        FOV_W (float): Vertical field of view in degrees.
        H (int): Height of the image in pixels.
        W (int): Width of the image in pixels.
        t_vec (np.ndarray): Translation vector of size (3,).
        r_vec (np.ndarray): Rotation vector of size (3,) in radians.
        small_std_xy (float): Standard deviation for small random errors.
        large_std_xy (float): Standard deviation for large random errors.
        p (float): Probability of applying large random error.

    Returns:
        tuple: (2D projections without errors, 2D projections with errors, indexes of points)
    """
    # Convert FOV to focal lengths
    focal_length_x = (W / 2) / np.tan(np.radians(FOV_H / 2))
    focal_length_y = (H / 2) / np.tan(np.radians(FOV_W / 2))

    # Rotation matrix from rotation vector
    rotation_matrix = R.from_rotvec(r_vec).as_matrix()

    # Transform points
    transformed_points = (rotation_matrix @ points_3d.T).T + t_vec

    # Project points into camera plane
    projected_points = transformed_points[:, :2] / transformed_points[:, 2, np.newaxis]
    projected_points[:, 0] = focal_length_x * projected_points[:, 0] + W / 2
    projected_points[:, 1] = focal_length_y * projected_points[:, 1] + H / 2

    # Add random errors
    errors = np.random.normal(0, small_std_xy, size=projected_points.shape)
    errors[:, 0] = errors[:, 0] * (1 - stabilities)
    errors[:, 1] = errors[:, 1] * (1 - stabilities)
    large_errors = np.random.normal(0, large_std_xy, size=projected_points.shape)
    apply_large_error = np.random.rand(len(projected_points)) < p
    errors[apply_large_error] = large_errors[apply_large_error]

    # Projected points with errors
    projected_points_with_errors = projected_points + errors

    return projected_points, projected_points_with_errors, np.arange(len(points_3d))

def visualize_projections_animation(H, W, projections, projections_with_errors, indexes, ax, colors):
    """Visualize 2D projections with and without errors for animation."""
    ax.clear()

    # Plot projections without errors (filled markers) and with errors (empty markers)
    for i, (proj, proj_err) in enumerate(zip(projections, projections_with_errors)):
        color = colors[i]  # Use consistent color for each point
        ax.scatter(proj[0], proj[1], c=[color], label=f"Point {i} (No Error)", marker="o", edgecolors="black")
        ax.scatter(
            proj_err[0],
            proj_err[1],
            facecolors="none",
            edgecolors=[color],
            label=f"Point {i} (With Error)",
            marker="o",
        )
        ax.text(proj[0], proj[1], str(i), color="black", fontsize=8)

    # Configure plot
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.invert_yaxis()
    ax.set_xlabel("Image Width (pixels)")
    ax.set_ylabel("Image Height (pixels)")
    ax.set_title("2D Projections of 3D Points")
    ax.legend(loc="upper right", fontsize=8, markerscale=0.6)

def simulate_frames(
    points_3d, FOV_H, FOV_W, H, W,
    init_t_vec, init_r_vec,
    initial_velocity_t_vec, initial_velocity_r_vec,
    initial_acceleration_t_vec, initial_acceleration_r_vec,
    small_std_xy, large_std_xy, p, num_frames, all_stabilities
):
    """Simulate and visualize multiple frames of projections."""
    t_vec = init_t_vec.astype(np.float64).copy()
    r_vec = init_r_vec.astype(np.float64).copy()
    velocity_t_vec = initial_velocity_t_vec.astype(np.float64).copy()
    velocity_r_vec = initial_velocity_r_vec.astype(np.float64).copy()

    all_projections = []
    all_projections_with_errors = []

    fig, ax = plt.subplots(figsize=(10, 10))

    # Generate consistent colors for all points
    colors = [np.random.rand(3,) for _ in range(len(points_3d))]

    for frame in range(num_frames):
        projections, projections_with_errors, indexes = project_points(
            points_3d, FOV_H, FOV_W, H, W, t_vec, r_vec, small_std_xy, large_std_xy, p, all_stabilities[frame]
        )

        all_projections.append(projections)
        all_projections_with_errors.append(projections_with_errors)

        # Visualize the current frame
        print(f"Frame {frame + 1}/{num_frames}")
        visualize_projections_animation(H, W, projections, projections_with_errors, indexes, ax, colors)

        plt.pause(SLEEP_PAUSE)  # Pause for 1 second between frames

        # Update translation and rotation vectors using velocity and acceleration
        velocity_t_vec += initial_acceleration_t_vec
        velocity_r_vec += initial_acceleration_r_vec
        t_vec += velocity_t_vec
        r_vec += velocity_r_vec

    plt.show()

    return all_projections, all_projections_with_errors


if __name__ == "__main__":
    K = 10
    points_3d = np.random.uniform(-1, 1, size=(K, 3)) + np.array([0, 0, 0])  # Ensure positive Z
    FOV_H = 90
    FOV_W = 60
    H = 1080
    W = 1920
    init_t_vec = np.array([0, 0, 5])
    init_r_vec = np.array([0, 0, 0])
    initial_velocity_t_vec = np.array([0.01, 0.01, 0.0])
    initial_velocity_r_vec = np.array([0.001, 0.010, 0.040])
    initial_acceleration_t_vec = np.array([0.0, -0.0001, 0.0001])
    initial_acceleration_r_vec = np.array([0.00, 0.00, 0.0])
    small_std_xy = 10
    large_std_xy = 20
    p = 0.1
    num_frames = 1000
    min_stability = 0.5
    max_stability = 1
    delta_stability = 0.05
    out_pickle_path = "stam.pickle"

    all_stabilities = gen_stabilities(min_stability,max_stability,delta_stability, num_frames, K)

    all_projections, all_projections_with_errors = simulate_frames(
        points_3d, FOV_H, FOV_W, H, W,
        init_t_vec, init_r_vec,
        initial_velocity_t_vec, initial_velocity_r_vec,
        initial_acceleration_t_vec, initial_acceleration_r_vec,
        small_std_xy, large_std_xy, p, num_frames,all_stabilities
    )

    #save all_projections, all_projections_with_errors, points_3d, FOV_H, FOV_W, H, W,
    out_dict = {
        "all_projections": all_projections,
        "all_projections_with_errors": all_projections_with_errors,
        "points_3d":points_3d,
        "FOV_H": FOV_H,
        "FOV_W": FOV_W,
        "H":H,
        "W":W
    }

    with open(out_pickle_path, "wb") as f:
        pickle.dump(out_dict, f)


    print('OK')