import numpy as np
from PIL import Image
import os

# --- Step 1: Set the Image Path ---
# Replace this path with your actual image file path
# Example: "C:/Users/YourUser/Pictures/photo.jpg"
image_file_path = "13.jpg"

# --- Step 2: Load the Image or Fallback to Dummy Image ---
if os.path.exists(image_file_path):
    try:
        original_image = Image.open(image_file_path)
        print(f"✅ Loaded Image from: {image_file_path}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        print("Creating a dummy image instead.")
        dummy_image_data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        original_image = Image.fromarray(dummy_image_data, 'RGB')
else:
    print("⚠️ Image file not found. Creating a dummy image.")
    dummy_image_data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    original_image = Image.fromarray(dummy_image_data, 'RGB')

# --- Step 3: Convert to RGB (in case it’s not) ---
original_image = original_image.convert('RGB')
original_array = np.array(original_image)
print(f"\nOriginal Image Size (H, W, Channels): {original_array.shape}")
print(f"Total RGB Features: {original_array.size}")

# --- Step 4: Convert to Grayscale ---
grayscale_image = original_image.convert('L')
grayscale_array = np.array(grayscale_image)
print(f"\n--- Grayscale Conversion ---")
print(f"Grayscale Image Size (H, W): {grayscale_array.shape}")
print(f"Total Grayscale Features: {grayscale_array.size}")

# --- Step 5: Flatten (Vectorize) the Pixels ---
flattened_features = grayscale_array.flatten()
print(f"\n--- Flattened Feature Vector ---")
print(f"Shape: {flattened_features.shape}")
print(f"First 10 Features: {flattened_features[:10]}")
print(f"Data Type: {flattened_features.dtype}")

# --- Optional: Display Images (if you want) ---
# from matplotlib import pyplot as plt
# plt.figure(figsize=(8,4))
# plt.subplot(1,2,1); plt.imshow(original_image); plt.title("Original Image"); plt.axis('off')
# plt.subplot(1,2,2); plt.imshow(grayscale_image, cmap='gray'); plt.title("Grayscale Image"); plt.axis('off')
# plt.show()
