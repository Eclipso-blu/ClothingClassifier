import tensorflow as tf
from tensorflow import keras
from tensorflow import image
from tensorflow import InceptionV3
from tensorflow import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt

# Load the InceptionV3 model pre-trained on ImageNet data
model = InceptionV3(weights='imagenet')

# Function to preprocess the image for prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Function to make predictions on the image
def predict_image(img_path):
    processed_img = preprocess_image(img_path)
    predictions = model.predict(processed_img)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i + 1}: {label} ({score:.2f})")

    # Display the image
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.show()

# Example usage
image_path = 'path/to/your/image.jpg'
predict_image(image_path)
