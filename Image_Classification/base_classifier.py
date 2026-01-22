import os
# Suppress oneDNN float32 warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

def load_and_preprocess(img_path, target_size=(224, 224)):
    """Load an image file and preprocess it for MobileNetV2."""
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x, img

def predict_image(img_path, save_plot=True):
    """Predict the class of an image and optionally save the plot."""
    x, img = load_and_preprocess(img_path)

    # Make prediction
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=3)[0]

    # Print predictions
    print("Predictions:")
    for label, name, prob in decoded:
        print(f"{name}: {prob:.4f}")

    # Plot image with top prediction as title
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Top: {decoded[0][1]} ({decoded[0][2]:.2f})")

    if save_plot:
        # Save the figure with a consistent filename
        plt.savefig('tiger_grancam_result.jpg')
        print("Saved plot as 'tiger_grancam_result.jpg'.")

# Example usage
if __name__ == "__main__":
    img_path = "images/Tiger.jpg"
    predict_image(img_path)


