import tensorflow as tf
import numpy as np
import cv2
import os
import time
from sklearn.metrics import f1_score, precision_score, recall_score
import lpips
import torch

# Load the models
classifier_model = tf.keras.models.load_model('pixelation_classifier_custom.h5')
restoration_model = tf.keras.models.load_model('image_restoration_model.h5')

# Initialize the LPIPS model
lpips_model = lpips.LPIPS(net='alex', verbose=False)

# Function to preprocess images
def preprocess_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path)
    original_size = image.shape[:2]
    image = cv2.resize(image, target_size)
    image = image / 255.0
    return image, original_size

# Function to restore images in batch
def restore_images_batch(images):
    restored_images = restoration_model.predict(images)
    return restored_images

# Function to calculate metrics
def calculate_metrics(true_labels, predictions):
    f1 = f1_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    return f1, precision, recall

# Function to calculate PSNR
def calculate_psnr(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

# Function to calculate PSNR and LPIPS
def calculate_restoration_metrics(true_images, restored_images):
    psnr_values = [calculate_psnr(true_images[i], restored_images[i]) for i in range(len(true_images))]
    
    # Convert to NumPy arrays
    true_images_numpy = np.array(true_images)
    restored_images_numpy = np.array(restored_images)
    
    # Convert to PyTorch tensors 
    true_images_torch = torch.from_numpy(true_images_numpy).permute(0, 3, 1, 2).float()
    restored_images_torch = torch.from_numpy(restored_images_numpy).permute(0, 3, 1, 2).float()
    
    # Calculate LPIPS
    lpips_values = lpips_model(true_images_torch, restored_images_torch)
    
    return np.mean(psnr_values), np.mean(lpips_values.detach().numpy())

# Inference on a batch of images
def run_inference(image_paths, true_paths):
    images = []
    true_images = []
    original_sizes = []
    for image_path, true_path in zip(image_paths, true_paths):
        img, original_size = preprocess_image(image_path)
        true_img, _ = preprocess_image(true_path)
        images.append(img)
        true_images.append(true_img)
        original_sizes.append(original_size)
    
    images = np.array(images)
    true_images = np.array(true_images)
    
    start_time = time.time()
    predictions = classifier_model.predict(images)
    end_time = time.time()
    
    # Convert predictions to binary
    predictions = (predictions > 0.5).astype(int).flatten()
    
    classification_fps = len(image_paths) / (end_time - start_time)
    
    # Restore images classified as pixelated in batch
    pixelated_images = images[predictions == 1]
    restoration_start_time = time.time()
    restored_images = restore_images_batch(pixelated_images)
    restoration_end_time = time.time()
    
    final_restored_images = np.array(images)
    final_restored_images[predictions == 1] = restored_images
    
    restoration_fps = len(pixelated_images) / (restoration_end_time - restoration_start_time)
    
    # Calculate restoration metrics
    psnr, lpips_value = calculate_restoration_metrics(true_images, final_restored_images)
    
    return predictions, classification_fps, restoration_fps, psnr, lpips_value, final_restored_images, original_sizes

# Example usage
image_folder = 'test/test_images' # replace with your test images path
ground_truth_folder = 'test/ground_truth' # replace with your ground truth path
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.png')]
true_paths = [os.path.join(ground_truth_folder, img) for img in os.listdir(ground_truth_folder) if img.endswith('.png')]
true_labels = [0 if 'non_pixelated' in img else 1 for img in image_paths]

predictions, classification_fps, restoration_fps, psnr, lpips_value, restored_images, original_sizes = run_inference(image_paths, true_paths)
f1, precision, recall = calculate_metrics(true_labels, predictions)

# Display predictions
class_names = {0: 'non_pixelated', 1: 'pixelated'}
for img_path, pred in zip(image_paths, predictions):
    print(f"{img_path}: {class_names[pred]}")

print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Classification FPS: {classification_fps}")
print(f"Restoration FPS: {restoration_fps}")
print(f"PSNR: {psnr}")
print(f"LPIPS: {lpips_value}")

# Save restored images in original sizes
os.makedirs('restored_images', exist_ok=True)
for img_path, restored_img, original_size in zip(image_paths, restored_images, original_sizes):
    restored_img = (restored_img * 255).astype(np.uint8)
    restored_img = cv2.resize(restored_img, original_size)
    restored_path = os.path.join('restored_images', os.path.basename(img_path))
    cv2.imwrite(restored_path, restored_img)