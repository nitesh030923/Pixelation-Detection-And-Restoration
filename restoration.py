import tensorflow as tf
from tensorflow.keras.layers import Input, DepthwiseConv2D, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Define path
train_dir = 'dataset/train'

# Function to load images from folder
def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

# Load training data
pixelated_images, pixelated_filenames = load_images_from_folder(os.path.join(train_dir, 'pixelated'))
ground_truth_images, _ = load_images_from_folder(os.path.join(train_dir, 'non_pixelated'))

# Resize and normalize images
def preprocess_images(images):
    resized_images = np.array([cv2.resize(img, (128, 128)) for img in images])
    normalized_images = resized_images / 255.0
    return normalized_images

pixelated_images = preprocess_images(pixelated_images)
ground_truth_images = preprocess_images(ground_truth_images)

# Split the data into training and validation sets
train_pixelated, val_pixelated, train_ground, val_ground = train_test_split(
    pixelated_images, ground_truth_images, test_size=0.2, random_state=42
)

# Custom PSNR metric
def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

# Define the simplified and efficient model
def create_efficient_model():
    inputs = Input(shape=(128, 128, 3))
    
    # Encoder (contracting path)
    conv1 = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, activation='relu')(inputs)
    conv1 = Conv2D(64, (1, 1), padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, activation='relu')(pool1)
    conv2 = Conv2D(128, (1, 1), padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Bottom of the model
    conv3 = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, activation='relu')(pool2)
    conv3 = Conv2D(256, (1, 1), padding='same', activation='relu')(conv3)
    
    # Decoder (expansive path)
    up4 = Concatenate()([UpSampling2D(size=(2, 2))(conv3), conv2])
    conv4 = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, activation='relu')(up4)
    conv4 = Conv2D(128, (1, 1), padding='same', activation='relu')(conv4)
    
    up5 = Concatenate()([UpSampling2D(size=(2, 2))(conv4), conv1])
    conv5 = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, activation='relu')(up5)
    conv5 = Conv2D(64, (1, 1), padding='same', activation='relu')(conv5)
    
    # Output layer
    outputs = Conv2D(3, (1, 1), activation='sigmoid')(conv5)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Instantiate and compile the model
efficient_model = create_efficient_model()
efficient_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=[psnr])

# Train the efficient model
efficient_model.fit(train_pixelated, train_ground, epochs=20, batch_size=16, validation_data=(val_pixelated, val_ground))

# Save the efficient model
efficient_model.save('image_restoration_model.h5')
