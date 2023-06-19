import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the input shape and batch size
input_shape = (256, 256, 3)
batch_size = 32

# Define the paths to your dataset folders
dataset_folder = 'D:/triplet_losss/train_dataset'
train_folder = os.path.join(dataset_folder, 'train')
val_folder = os.path.join(dataset_folder, 'val')

# Create an ImageDataGenerator for data preprocessing and augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,  # 20% of the data will be used for validation
)

# Generate training data from the "train" folder
train_data = datagen.flow_from_directory(
    train_folder,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary',
    subset='training',  # Use the training subset of the data
    shuffle=True,
)

# Generate validation data from the "train" folder
val_data = datagen.flow_from_directory(
    train_folder,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',  # Use the validation subset of the data
    shuffle=True,
)

# Load the MobileNet model with pre-trained weights
mobilenet_model = MobileNet(input_shape=input_shape, include_top=False, weights='imagenet')

# Freeze the layers of the pre-trained model
for layer in mobilenet_model.layers:
    layer.trainable = False

# Add a new classification head on top of the pre-trained model
model = tf.keras.models.Sequential([
    mobilenet_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with an optimizer and loss function
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
model.fit(train_data, validation_data=val_data, epochs=epochs)

# Save the trained model
model.save('D:/triplet_losss/weights')
