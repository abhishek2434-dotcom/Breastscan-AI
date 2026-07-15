%%writefile mask_extraction.py
import os
import numpy as np
from PIL import Image
import cv2 # OpenCV for image processing
import matplotlib.pyplot as plt

# Define paths (consistent with main_pipeline)
DATASET_PATH = '/content/Dataset_BUSI_with_GT'
OUTPUT_MASK_PATH = '/content/extracted_masks'

def load_grayscale(img_path):
    return np.array(Image.open(img_path).convert('L'))

def normalize_image(img):
    img = img.astype(np.float64)
    img_min, img_max = img.min(), img.max()
    if img_max - img_min == 0:
        return img.astype(np.uint8)
    return ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)

def extract_mask_from_image(img_path, output_dir):
    # Load the original image
    img_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_original is None:
        print(f"Error: Could not load image at {img_path}")
        return

    # Load the corresponding mask image
    mask_img_path = img_path.replace('.png', '_mask.png').replace('.jpg', '_mask.jpg').replace('.jpeg', '_mask.jpeg')
    img_mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)

    if img_mask is None:
        # If no explicit mask file exists, attempt simple thresholding or contour detection
        # This is a fallback for datasets without explicit mask files
        print(f"Warning: Mask file not found for {os.path.basename(img_path)}. Attempting automated mask extraction.")
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(img_original, (5, 5), 0)
        
        # Use Otsu's thresholding to get a binary image
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create an empty mask and draw the largest contour
        auto_mask = np.zeros_like(img_original)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(auto_mask, [largest_contour], -1, 255, cv2.FILLED)
        img_mask = auto_mask

    # Ensure mask is binary (0 or 255)
    _, binary_mask = cv2.threshold(img_mask, 127, 255, cv2.THRESH_BINARY)

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the extracted mask
    mask_filename = os.path.basename(img_path).replace('.png', '_extracted_mask.png')
    output_mask_path = os.path.join(output_dir, mask_filename)
    cv2.imwrite(output_mask_path, binary_mask)
    return output_mask_path

def process_all_masks(dataset_path=DATASET_PATH, output_path=OUTPUT_MASK_PATH):
    print(f"--- Starting mask extraction to {output_path} ---")
    classes = {'normal', 'benign', 'malignant'}
    extracted_count = 0

    for class_name in classes:
        folder = os.path.join(dataset_path, class_name)
        if not os.path.exists(folder):
            print(f"  WARNING: folder not found: {folder}")
            continue

        files_in_folder = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for fname in files_in_folder:
            if '_mask' not in fname.lower(): # Only process original images, not existing masks
                img_path = os.path.join(folder, fname)
                mask_output_path = extract_mask_from_image(img_path, output_path)
                if mask_output_path:
                    extracted_count += 1
    print(f"--- Finished mask extraction. Total {extracted_count} masks extracted/generated. ---")

def visualize_mask_extraction(original_img_path, mask_output_path):
    # Load images
    original_img = cv2.imread(original_img_path, cv2.IMREAD_GRAYSCALE)
    extracted_mask = cv2.imread(mask_output_path, cv2.IMREAD_GRAYSCALE)

    if original_img is None or extracted_mask is None:
        print("Error: Could not load images for visualization.")
        return

    # Create a simple overlay (optional: for better visualization)
    # Convert mask to color (red)
    mask_color = cv2.cvtColor(extracted_mask, cv2.COLOR_GRAY2BGR)
    mask_color[:, :, 0] = 0 # Blue channel to 0
    mask_color[:, :, 1] = 0 # Green channel to 0
    mask_color[:, :, 2] = extracted_mask # Red channel from mask

    # Convert original to BGR for overlay
    original_bgr = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

    # Blend images (e.g., 50% original, 50% mask)
    overlay = cv2.addWeighted(original_bgr, 0.7, mask_color, 0.3, 0)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(extracted_mask, cmap='gray')
    plt.title('Extracted Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)) # Matplotlib expects RGB
    plt.title('Original with Mask Overlay')
    plt.axis('off')
    plt.show()