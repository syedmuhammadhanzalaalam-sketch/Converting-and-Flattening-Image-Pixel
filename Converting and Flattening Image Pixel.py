import numpy as np 
from PIL import Image 
import os # To check if a file exists 
# --- Configuration for Image Loading --- 
# IMPORTANT:
# Set 'image_file_path' below to the actual path of your image file if you want to load your own image.
# Examples:
#   image_file_path = 'my_image.png'  # For an image in the same folder as this script
#   image_file_path = 'C:/Users/YourUser/Pictures/my_photo.jpg'  # On Windows
#   image_file_path = '/home/youruser/images/my_photo.png'       # On Linux/macOS
# Set to None to use a randomly generated dummy image instead.
image_file_path = None
 
# 1. Load Image or Create a Dummy Image 
original_image = None 
if image_file_path and os.path.exists(image_file_path): 
    try: 
        original_image = Image.open(image_file_path) 
        print(f"--- Loaded Image from File: {image_file_path} ---") 
    except Exception as e: 
        print(f"Error loading image from {image_file_path}: {e}") 
        print("Creating a dummy image instead.") 
        # Fallback to dummy image if file loading fails 
        dummy_image_data = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8) 
        original_image = Image.fromarray(dummy_image_data, 'RGB') 
        print(f"--- Created Dummy Image (100x100 RGB) ---") 
else: 
    print("No valid image file path provided or file not found.") 
    print("Creating a dummy image instead.") 
    # Create a simple 100x100 RGB Image if no valid path is given 
    dummy_image_data = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8) 
    original_image = Image.fromarray(dummy_image_data, 'RGB') 
    print(f"--- Created Dummy Image (100x100 RGB) ---") 
 
# Ensure the image is in RGB format for consistent processing if it wasn't already 
original_image = original_image.convert('RGB') 
original_image_array = np.array(original_image) 
 
print(f"Original Image Size (H, W, Channels): {original_image_array.shape}") 
print(f"Total Features (Pixels) in RGB: {original_image_array.shape[0] * 
original_image_array.shape[1] * original_image_array.shape[2]}") 
 
# Display the original image (optional, requires matplotlib but good for visualization) 
# from matplotlib import pyplot as plt 
# plt.imshow(original_image) 
# plt.title("Original Image") 
# plt.axis('off') 
# plt.show() 
 
 
# 2. Apply Simple Feature Engineering: Grayscale Conversion 
# Grayscale is a simple preprocessing/feature reduction technique. 
grayscale_image = original_image.convert('L') # 'L' mode is for Grayscale 
grayscale_array = np.array(grayscale_image) 
 
print(f"\n--- Grayscale Conversion (Feature Reduction) ---") 
print(f"Grayscale Image Size (H, W, Channels): {grayscale_array.shape}") 
print(f"Total Features (Pixels) after Grayscale: {grayscale_array.shape[0] * 
grayscale_array.shape[1]}") 
# Display the grayscale image (optional) 
# plt.imshow(grayscale_image, cmap='gray') 
# plt.title("Grayscale Image") 
# plt.axis('off') 
# plt.show() 
# 3. Apply Simple Feature Engineering: Pixel Flattening 
# Flattening converts the 2D (or 3D) matrix into a 1D feature vector 
# for input into a traditional ML algorithm. 
flattened_features = grayscale_array.flatten() 
print(f"\n--- Pixel Flattening (Vectorization) ---") 
print(f"Flattened Feature Vector Shape: {flattened_features.shape}") 
print("\nFirst 10 Features (Pixel Values) in Machine Understandable Format:") 
print(flattened_features[:10]) 
# Verify the type - it's a numerical array, ready for an ML model 
print(f"\nData Type of Final Features: {flattened_features.dtype}")