import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
import os

# Define path
train_dir = 'dataset/train'

# Image data generators
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split the data into training and validation sets
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)
# Load the pre-trained MobileNetV2 model + higher level layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze the first 100 layers of the model
for layer in base_model.layers[:100]:
    layer.trainable = False
# Simplified CNN architecture
def create_model():
    model = Sequential([
         base_model,
         GlobalAveragePooling2D(),
         Dense(128, activation='relu'),
         Dropout(0.5),
         Dense(1, activation='sigmoid')
    ])
    
    return model

# Instantiate and compile the model
model = create_model()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator
)

# Save the model
model.save('pixelation_classifier_custom.h5')
