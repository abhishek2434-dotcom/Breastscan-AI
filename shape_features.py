%%writefile shape_features.py
import numpy as np
from skimage.measure import regionprops
from skimage.morphology import binary_erosion, binary_dilation

def extract_shape_features(mask):
    """
    Extracts shape-based features from a binary mask.

    Args:
        mask (np.array): A 2D binary numpy array representing the tumor mask.
                         Assumes 255 for foreground, 0 for background.

    Returns:
        list: A list of extracted shape features.
    """
    if mask.sum() == 0: # Handle empty mask case
        return [0.0] * 10 # Return 10 zeros if no mask detected

    # Ensure mask is boolean for regionprops
    binary_mask = mask > 0
    
    # Label connected components to handle multiple regions if present
    # We typically assume a single main tumor region for feature extraction
    labeled_mask = np.array(binary_mask, dtype=int)
    from scipy.ndimage import label
    labeled_array, num_features = label(labeled_mask)
    
    if num_features == 0: # Still no features after labeling
        return [0.0] * 10

    # Get properties of the largest region (assuming it's the tumor)
    props = regionprops(labeled_array)
    if not props:
        return [0.0] * 10
        
    # Find the largest region by area
    largest_prop = max(props, key=lambda p: p.area)

    area = float(largest_prop.area)
    perimeter = float(largest_prop.perimeter)
    
    # Handle cases where perimeter might be zero or very small for tiny regions
    if perimeter == 0:
        circularity = 0.0
    else:
        circularity = float(4 * np.pi * area / (perimeter ** 2))

    eccentricity = float(largest_prop.eccentricity)
    solidity = float(largest_prop.solidity)
    extent = float(largest_prop.extent)
    
    # Orientation: angle between the x-axis and the major axis of the ellipse
    # Fits to the object, in radians between -pi/2 and pi/2.
    orientation = float(largest_prop.orientation) # Radians

    major_axis_length = float(largest_prop.major_axis_length)
    minor_axis_length = float(largest_prop.minor_axis_length)

    # Ratio of major to minor axis length
    if minor_axis_length == 0:
        aspect_ratio = 0.0 # Avoid division by zero
    else:
        aspect_ratio = float(major_axis_length / minor_axis_length)

    # Convex area (area of the convex hull)
    convex_area = float(largest_prop.convex_area)

    # Erosion and Dilation for roughness/smoothness indicators
    eroded_mask = binary_erosion(binary_mask, selem=np.ones((3,3)))
    dilated_mask = binary_dilation(binary_mask, selem=np.ones((3,3)))
    
    eroded_area = float(eroded_mask.sum())
    dilated_area = float(dilated_mask.sum())

    return [
        area,
        perimeter,
        circularity,
        eccentricity,
        solidity,
        extent,
        orientation,
        aspect_ratio,
        convex_area,
        eroded_area / area if area > 0 else 0.0 # Ratio of eroded area to original area
    ]