import PIL
import json, os, copy
import numpy as np
import tkinter as tk
from PIL import Image

import numpy as np
from scipy.ndimage import correlate
from scipy.spatial import distance
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def find_default_params_2d2d(im1,im2,markers1,markers2,factor=1):

    max_size = max(np.array(markers1)[:,0].max() - np.array(markers1)[:,0].min(), np.array(markers1)[:,1].max() - np.array(markers1)[:,1].min())

    search_radius = int(max_size*0.75*factor)
    small_search_rad =  int(search_radius*0.5)
    mask_radius = int(max_size*0.2)

    return(search_radius,small_search_rad,mask_radius)


def visualize_markers(image, markers, marker_color='red', marker_size=50, show_coordinates=False, title = None):
    """
    Visualize markers on top of a grayscale image.

    Parameters:
    image: 2D array (grayscale image).
    markers: List of 2D marker coordinates [(x1, y1), (x2, y2), ...].
    marker_color: Color of the markers (default: 'red').
    marker_size: Size of the markers (default: 50).
    show_coordinates: Boolean to display marker coordinates as text (default: False).

    Returns:
    None. Displays the image with markers overlaid.
    """
    if(title is not None):
        plt.figure(title)
    # Display the grayscale image
    plt.imshow(image, cmap='gray')

    # Overlay the markers
    for i, (x, y) in enumerate(markers):
        plt.scatter([x], [y], c=marker_color, s=marker_size)
        if show_coordinates:
            plt.text(x, y, f'({x}, {y})', color=marker_color, fontsize=10, ha='right', va='bottom')

    plt.axis('off')  # Turn off axis labels
    plt.show()



def compute_mask(im1, markers1, maskRadius):
    """
    Compute a binary mask where pixels are 1 if they are within maskRadius of the nearest marker.

    Parameters:
    im1: Grayscale image (2D array).
    markers1: List of 2D marker coordinates (x, y).
    maskRadius: The radius around each marker for the mask.

    Returns:
    Binary mask (same size as im1) where pixels within maskRadius of a marker are 1, and others are 0.
    """
    # Initialize mask as a zero array of the same size as im1
    mask = np.zeros(im1.shape, dtype=np.uint8)

    # Create a 2D grid of pixel coordinates
    rows, cols = np.indices(im1.shape)
    pixel_coords = np.stack((rows, cols), axis=-1)  # Shape (height, width, 2)

    # Iterate over each marker and set mask values to 1 within the maskRadius
    for marker in markers1:
        # Compute the Euclidean distance from this marker to all pixels
        distances = np.sqrt((pixel_coords[..., 0] - marker[1]) ** 2 + (pixel_coords[..., 1] - marker[0]) ** 2)

        # Set mask to 1 where distance is less than or equal to maskRadius
        mask[distances <= maskRadius] = 1

    return mask


def normalize(image):
    """Normalize the input image to have zero mean and unit variance."""
    mean = np.mean(image)
    std = np.std(image)
    if std == 0:
        return image - mean
    return (image - mean) / std

def my_normalize(image,mask):
    """Normalize the input image to have zero mean and unit variance."""
    maskbool = mask.astype(bool)
    masked_pixels = image[maskbool]

    mean = np.mean(masked_pixels)

    std = np.std(masked_pixels)
    if std == 0:
        return masked_pixels - mean
    return (masked_pixels - mean) / std


def compute_ncc(patch1, patch2):
    """Compute the normalized cross-correlation (NCC) between two patches."""
    norm_patch1 = normalize(patch1)
    norm_patch2 = normalize(patch2)
    return np.sum(norm_patch1 * norm_patch2)

def my_compute_ncc(patch1, patch2, mask1):
    norm_patch1 = my_normalize(patch1, mask1)
    norm_patch2 = my_normalize(patch2, mask1)
    return np.mean(norm_patch1 * norm_patch2)

def cut_out_roi(roi_min_row,roi_max_row, roi_min_col, roi_max_col,imshape,full_mask , min_mask_factor = 0.5):

    roi_min_row_orig, roi_max_row_orig, roi_min_col_orig, roi_max_col_orig = (roi_min_row, roi_max_row, roi_min_col, roi_max_col)
    roi_min_row_delta = roi_max_row_delta =roi_min_col_delta =roi_max_col_delta = 0

    # if roi_min_row < 0 or roi_max_row >= im2.shape[0] or roi_min_col < 0 or roi_max_col >= im2.shape[1]:
    if(roi_min_row < 0):
        roi_min_row=0
    if (roi_min_col < 0):
        roi_min_col = 0
    if (roi_max_row >= imshape[0]):
        roi_max_row = imshape[0] -1
    if (roi_max_col >= imshape[1]):
        roi_max_col = imshape[1] -1

    WM = np.sum(full_mask)
    if(roi_min_col < roi_max_col and roi_min_row < roi_max_row):
        small_mask = full_mask[roi_min_row:roi_max_row + 1, roi_min_col:roi_max_col + 1]
        WSM = np.sum(small_mask)
        if(WSM/WM<min_mask_factor ):
            roi_min_row_delta = roi_max_row_delta =  roi_min_col_delta = roi_max_col_delta = None
        else:
            roi_min_row_delta, roi_max_row_delta, roi_min_col_delta, roi_max_col_delta = (
            roi_min_row-roi_min_row_orig, roi_max_row-roi_max_row_orig, roi_min_col-roi_min_col_orig, roi_max_col-roi_max_col_orig)
    else:
        roi_min_row_delta = roi_max_row_delta =  roi_min_col_delta = roi_max_col_delta = None

    return roi_min_row_delta ,roi_max_row_delta,roi_min_col_delta,roi_max_col_delta


def overlay_2D_to_2D(im1, im2, markers1, markers2, mask1, searchRadius, thresh = 0, show = False,
                     around_current = True):
    """
    Find the optimal shift between two images based on normalized cross-correlation.

    Parameters:
    im1, im2: Grayscale images of the same size.
    markers1, markers2: Two sets of 2D markers (lists of (x, y) coordinates) corresponding to im1 and im2.
    mask1: Binary mask for the target region in the first image.
    searchRadius: Maximum search radius for the optimal shift.

    Returns:
    Optimal shift (dx, dy).
    """

    if(markers2 is None):
        markers2 = markers1.copy()

    # Get the bounding box of the target region in the first image
    target_region = np.where(mask1)
    min_row, max_row = np.min(target_region[0]), np.max(target_region[0])
    min_col, max_col = np.min(target_region[1]), np.max(target_region[1])

    # Extract the target patch from the first image using the mask
    target_patch = im1[min_row:max_row + 1, min_col:max_col + 1] * mask1[min_row:max_row + 1, min_col:max_col + 1]
    target_mask =  mask1[min_row:max_row + 1, min_col:max_col + 1] * mask1[min_row:max_row + 1, min_col:max_col + 1]

    # Initialize the best correlation and shift
    best_correlation = -np.inf
    best_shift = (0, 0)
    best_patch2 = None

    center_of_mass_markers1 = np.average(markers1,axis =0)
    center_of_mass_markers2 = np.average(markers2,axis=0)

    y_shift,x_shift = center_of_mass_markers2 - center_of_mass_markers1
    if(around_current == False):
        x_shift, y_shift = (0,0) # serach around previous

    # Search for the optimal shift within the given radius
    for dx in range(-searchRadius + int(x_shift), searchRadius +  int(x_shift) + 1):
        for dy in range(-searchRadius + int(y_shift), searchRadius + int(y_shift) + 1):
            # Shift the second image's markers
            shifted_markers2 = [(x + dx, y + dy) for (x, y) in markers2]

            # Calculate the region of interest in the second image
            roi_min_row = min_row + dx
            roi_max_row = max_row + dx
            roi_min_col = min_col + dy
            roi_max_col = max_col + dy

            roi_min_row_delta, roi_max_row_delta, roi_min_col_delta, roi_max_col_delta = (0,0,0,0)

            if roi_min_row < 0 or roi_max_row >= im2.shape[0] or roi_min_col < 0 or roi_max_col >= im2.shape[1]:
                #continue
                roi_min_row_delta,roi_max_row_delta, roi_min_col_delta, roi_max_col_delta = cut_out_roi(roi_min_row,roi_max_row, roi_min_col, roi_max_col,imshape = im2.shape,full_mask = mask1, min_mask_factor = 0.5)
                if(roi_min_row_delta is None):
                    continue  # Skip out-of-bounds shifts
                # else:
                #     print("cut_out worked")

            # Extract the target patch from the first image using the mask
            target_patch = im1[min_row+roi_min_row_delta:max_row+roi_max_row_delta + 1,
                           min_col + roi_min_col_delta:max_col + roi_max_col_delta + 1] \
                           * mask1[min_row+roi_min_row_delta:max_row + roi_max_row_delta + 1,
                             min_col + roi_min_col_delta:max_col + roi_max_col_delta + 1]
            target_mask = mask1[min_row+roi_min_row_delta:max_row+roi_max_row_delta + 1,
                          min_col + roi_min_col_delta:max_col + roi_max_col_delta + 1]

            #try:
            # Extract the patch from the second image
            patch2 = im2[roi_min_row + roi_min_row_delta:roi_max_row + roi_max_row_delta + 1, roi_min_col + roi_min_col_delta:roi_max_col + roi_max_col_delta + 1]*target_mask
            # except:
            #     print('ooops')

            # Compute normalized cross-correlation
            #ncc = compute_ncc(target_patch, patch2)

            ncc = my_compute_ncc(target_patch,patch2, target_mask)
            # Update the best correlation and shift
            if ncc > best_correlation:
                best_correlation = ncc
                best_shift = (dx, dy)
                best_patch2 = patch2.copy()
    if(show):
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(target_patch)
        axs[0].set_title('Target')
        if(best_patch2 is not None):
            axs[1].imshow(best_patch2)
            axs[1].set_title('Best fit, corr' + str(best_correlation))
            plt.show()

    #find best shifted_markers1, best_shifted_marker2
    best_shifted_markers1 = markers1.copy()
    best_shifted_markers2 = markers2.copy()
    print("best_correlation = " + str(best_correlation))
    if(best_correlation > thresh):
        for mi, ma1 in enumerate(best_shifted_markers1):
            best_shifted_markers1[mi][0] = best_shifted_markers1[mi][0] +  best_shift[1]
            best_shifted_markers1[mi][1] = best_shifted_markers1[mi][1] + best_shift[0]

            best_shifted_markers2[mi][0] = best_shifted_markers2[mi][0] + best_shift[1] - y_shift
            best_shifted_markers2[mi][1] = best_shifted_markers2[mi][1] + best_shift[0] - x_shift



    return best_shift, best_shifted_markers1, best_shifted_markers2


def refine_overlay_2D_to_2D(im1, im2, markers1, best_shifted_markers1, best_shifted_markers2,
                            refineMaskRad, searchRadius, thresh = 0, show = False, around_current = True):
    NM = len(markers1)
    best_shifted_markers2_refined = copy.deepcopy(best_shifted_markers2)
    best_shifted_markers1_refined = copy.deepcopy(best_shifted_markers1)

    for mai in range(NM):
        # refine location of each marker
        temp_markers_1 = [markers1[mai]]
        if(around_current):
            temp_markers_2 = [best_shifted_markers2[mai]]
        else:
            temp_markers_2 = [best_shifted_markers1[mai]]
        temp_mask1 = compute_mask(im1=im1, markers1=temp_markers_1, maskRadius=refineMaskRad)

        # find local shift
        local_best_shift, local_best_shifted_markers1, local_best_shifted_markers2 = overlay_2D_to_2D(
            im1=im1, im2=im2,
            markers1=temp_markers_1,
            markers2=temp_markers_2,
            mask1=temp_mask1,
            searchRadius=searchRadius,
            thresh=thresh,
            show = show,
            around_current = True) # ???

        best_shifted_markers2_refined[mai] = local_best_shifted_markers2[0]
        best_shifted_markers1_refined[mai] = local_best_shifted_markers1[0]

    return (best_shifted_markers1_refined, best_shifted_markers2_refined)


def save_data_2_json(data, data3D, out_json):
    #TODO
    pass

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0],rgb[1],rgb[2])

def pil_image_translate(img,x,y):
    a = 1
    b = 0
    c = x  # left/right (i.e. 5/-5)
    d = 0
    e = 1
    f = y  # up/down (i.e. 5/-5)
    img = img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f))
    return img

class CustomTooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.tooltip_window = None
    def enter(self, event=None):
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tooltip_window, text=self.text, justify='left',
                         background='#ffffff', relief='solid', borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)
    def leave(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()

