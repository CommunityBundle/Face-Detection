import violajones.IntegralImage as ii
import violajones.AdaBoost as ab
import violajones.Utils as utils
import pickle
import random
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import cv2


def load_image(path):
    """Load a single image, resize to 19x19, convert to grayscale, and normalize."""
    img = Image.open(path).convert('L')  # Convert to grayscale ('L' mode)
    img = img.resize((19, 19), Image.Resampling.LANCZOS)  # Resize to 19x19
    
    img_arr = np.array(img, dtype=np.float64)
    
    # Normalize image to range [0, 1]
    img_arr /= img_arr.max()
    
    return img_arr  # Return as numpy array

def load_image_convert(path):
    """Load a single image, convert to grayscale, save as PGM, and normalize."""
    img = Image.open(path).convert('L')  # Convert to grayscale ('L' mode)
    
    # Save as PGM file for consistency
    pgm_path = os.path.splitext(path)[0] + ".pgm"
    img.save(pgm_path, "PPM")  # PGM is under the PPM format in Pillow
    
    img_arr = np.array(img, dtype=np.float64)
    
    # Normalize image to range [0, 1]
    img_arr /= img_arr.max()
    
    return img_arr  # Return as numpy array

def detect_face_in_image(image_path, classifier_model_path='classifier_model.pkl'):
    """
    Detects if a face exists in the given image using a trained Viola-Jones classifier.

    :param image_path: Path to the random image
    :param classifier_model_path: Path to the saved classifier model
    """
    # Load the trained classifier
    with open(classifier_model_path, 'rb') as f:
        classifiers = pickle.load(f)

    # Load the image and convert it to grayscale
    img = load_image(image_path)
    if img is None:
        print("Error: Image could not be loaded.")
        return

    # Convert the grayscale image to an integral image
    integral_img = ii.to_integral_image(img)

    # Run the ensemble classifier
    is_face = utils.ensemble_vote(integral_img, classifiers)

    # Display the result
    result = "Face Detected!" if is_face else "No Face Detected!"
    print(result)

    # Show the image with the result
    plt.imshow(img, cmap='gray')
    plt.title(result)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Path to a random image
    random_image_path = "Image/Scenery.jpg"
    detect_face_in_image(random_image_path)
