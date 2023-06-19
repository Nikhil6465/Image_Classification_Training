import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('D:/triplet_losss/weights')

# Input and output folders
input_folder = 'cropped_images'
shoe_output_folder = 'siamese_data'
other_output_folder = 'other_data'

# Iterate over images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load and preprocess the input image
        image_path = os.path.join(input_folder, filename)
        image = load_img(image_path, target_size=(256, 256))
        image_array = img_to_array(image)
        image_array = image_array / 255.0  # Normalize pixel values between 0 and 1
        image_array = np.expand_dims(image_array, axis=0)  # Add a batch dimension

        # Make predictions
        predictions = model.predict(image_array)
        if predictions[0] < 0.5:
            output_folder = other_output_folder
        else:
            output_folder = shoe_output_folder

        # Save the image to the appropriate output folder
        output_path = os.path.join(output_folder, filename)
        image.save(output_path)
