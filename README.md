# Image Classification and Restoration Pipeline

This repository contains scripts for image classification and restoration using deep learning models, implemented with TensorFlow/Keras and OpenCV.

## Overview

The pipeline consists of two main tasks:

1. **Image Classification**: Determines whether an image is pixelated or non-pixelated.
2. **Image Restoration**: Restores pixelated images using a custom deep learning model.

## Setup

### Environment Setup

Ensure you have the following dependencies installed:
- TensorFlow
- NumPy
- OpenCV (cv2)
- scikit-learn (for metrics)
- lpips (perceptual similarity metric)
- PyTorch (for LPIPS calculations)

You can install these dependencies using pip:

```bash
pip install tensorflow opencv-python numpy scikit-learn lpips torch

```
### Model Files

- pixelation_classifier_custom.h5: Trained model for image classification (non_pixelated vs pixelated).
- image_restoration_model.h5: Trained model for image restoration.

### Data Preparation

Organize your dataset as follows:

- Training images for image classification should be in dataset/train directory, with subdirectories pixelated and non_pixelated.

- Test images for inference should be in test/test_images directory, with corresponding ground truth images in test/ground_truth directory.

- The inference is made to run for .png images if you have the images in any other format, please change the extension in the script (in line no. 104 and 105).

### Usage
1- Run Inference Script: Use the provided script to classify and restore images. (Update the path of test images folder)

```bash
python inference.py
```
2- Output:

- Predictions for each image (classified as pixelated or non_pixelated).
- Metrics such as F1 score, precision, recall, PSNR, and LPIPS.
- Restored images saved in the restored_images folder.

3- Interpreting Results:

- Classification Metrics: Assess accuracy and performance of the image classifier.
- Restoration Metrics: Evaluate quality of restored images using PSNR and perceptual similarity (LPIPS).

