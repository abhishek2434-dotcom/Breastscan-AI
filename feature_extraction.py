%%writefile feature_extraction.py
import numpy as np
from PIL import Image
import os
from scipy.stats import skew, kurtosis
from scipy import fft
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops
from skimage.morphology import binary_erosion, binary_dilation

# Assuming these are available from mask_extraction.py and shape_features.py
# or are common utility functions.
# For demonstration, including them directly or importing them if they were separate files.

def load_grayscale(img_path):
    """
    Loads an image in grayscale.
    """
    return np.array(Image.open(img_path).convert('L'))

def normalize_image(img):
    """
    Normalizes image pixel values to 0-255 range.
    """
    img = img.astype(np.float64)
    img_min, img_max = img.min(), img.max()
    if img_max - img_min == 0:
        return img.astype(np.uint8)
    return ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)

def extract_texture_features(img):
    """
    Extracts texture-based features (statistical and GLCM) from an image.
    """
    mean_intensity = float(np.mean(img))
    std_dev = float(np.std(img))
    variance = float(np.var(img))

    if std_dev == 0:
        skewness, kurt = 0.0, 0.0
    else:
        skewness = float(skew(img.flatten()))
        kurt = float(kurtosis(img.flatten()))

    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    hist_norm = hist / hist.sum()
    hist_norm = hist_norm[hist_norm > 0]
    shannon_entropy = float(-np.sum(hist_norm * np.log2(hist_norm))) if len(hist_norm) > 0 else 0.0

    fft_result = fft.fft2(img)
    fft_power = np.abs(fft_result) ** 2
    spectral_energy = float(np.sum(fft_power))

    freqs_y = fft.fftfreq(img.shape[0])
    freqs_x = fft.fftfreq(img.shape[1])
    fx, fy = np.meshgrid(freqs_x, freqs_y)
    freq_grid = np.sqrt(fx ** 2 + fy ** 2)

    spectral_centroid = 0.0 if spectral_energy == 0 else float(
        np.sum(freq_grid * fft_power) / spectral_energy)
    dominant_freq = float(freq_grid.flatten()[np.argmax(fft_power.flatten())])

    glcm = graycomatrix(img, distances=[5], angles=[0], levels=256,
                         symmetric=True, normed=True)
    glcm_contrast = float(graycoprops(glcm, 'contrast')[0, 0])
    glcm_homogeneity = float(graycoprops(glcm, 'homogeneity')[0, 0])
    glcm_correlation = float(graycoprops(glcm, 'correlation')[0, 0])
    glcm_energy = float(graycoprops(glcm, 'energy')[0, 0])

    return [
        mean_intensity, std_dev, variance, skewness, kurt,
        shannon_entropy, spectral_energy, spectral_centroid,
        dominant_freq, glcm_contrast, glcm_homogeneity,
        glcm_correlation, glcm_energy
    ]

def extract_shape_features(mask):
    """
    Extracts shape-based features from a binary mask.
    """
    if mask.sum() == 0: # Handle empty mask case
        return [0.0] * 10 # Return 10 zeros if no mask detected

    binary_mask = mask > 0
    from scipy.ndimage import label
    labeled_array, num_features = label(binary_mask)

    if num_features == 0: # Still no features after labeling
        return [0.0] * 10

    props = regionprops(labeled_array)
    if not props:
        return [0.0] * 10

    largest_prop = max(props, key=lambda p: p.area)

    area = float(largest_prop.area)
    perimeter = float(largest_prop.perimeter)

    circularity = 0.0 if perimeter == 0 else float(4 * np.pi * area / (perimeter ** 2))
    eccentricity = float(largest_prop.eccentricity)
    solidity = float(largest_prop.solidity)
    extent = float(largest_prop.extent)
    orientation = float(largest_prop.orientation)
    major_axis_length = float(largest_prop.major_axis_length)
    minor_axis_length = float(largest_prop.minor_axis_length)

    aspect_ratio = 0.0 if minor_axis_length == 0 else float(major_axis_length / minor_axis_length)
    convex_area = float(largest_prop.convex_area)

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
        eroded_area / area if area > 0 else 0.0
    ]

def extract_all_features(image_path, mask_path=None):
    """
    Extracts a comprehensive set of features (texture and shape) from an image.

    Args:
        image_path (str): Path to the original grayscale image.
        mask_path (str, optional): Path to the binary mask image. If None, only texture features are extracted.

    Returns:
        list: A concatenated list of all extracted features.
    """
    img = load_grayscale(image_path)
    normalized_img = normalize_image(img)

    texture_feats = extract_texture_features(normalized_img)

    shape_feats = []
    if mask_path and os.path.exists(mask_path):
        mask = load_grayscale(mask_path) > 0 # Convert to boolean mask
        shape_feats = extract_shape_features(mask)
    else:
        # If no mask or mask not found, return zeros for shape features
        # Number of shape features should be consistent with extract_shape_features output
        print(f"Warning: Mask not found for {os.path.basename(image_path)}. Returning zeros for shape features.")
        shape_feats = [0.0] * 10 # Assuming 10 shape features from extract_shape_features

    return texture_feats + shape_feats