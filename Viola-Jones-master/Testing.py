import violajones.IntegralImage as ii
import violajones.AdaBoost as ab
import violajones.Utils as utils
import pickle
import random
import os
from PIL import Image
import numpy as np

from PIL import Image, ImageOps

def load_image(path):
    """Load a single image, resize to 19x19, convert to grayscale, and normalize."""
    img = Image.open(path).convert('L')  # Convert to grayscale ('L' mode)
    img = img.resize((24, 24), Image.Resampling.LANCZOS)  # Resize to 19x19
    
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

def detect_faces(image, classifiers):
    """Detect faces by sliding the window and using the classifiers."""
    detected_faces = 0
    window_size = (24, 24)  # Typically, the window size for face detection
    step_size = 8  # The step size for sliding the window

    # Slide the window over the image
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            window = image[y:y + window_size[1], x:x + window_size[0]]
            integral_image = ii.to_integral_image(window)  # Convert window to integral image
            
            # Use the classifier to predict if the window contains a face
            if utils.ensemble_vote_all([integral_image], classifiers)[0]:
                detected_faces += 1  # Count this as a detected face

    return detected_faces

def test_random_image(classifiers, image_path):
    """Test a random image with the trained classifiers."""
    image = load_image(image_path)  # Load image as a numpy array
    detected_faces = detect_faces(image, classifiers)  # Count faces using sliding window

    return detected_faces

def main():
    pos_training_path = 'face.train/train/face'
    neg_training_path = 'face.train/train/non-face'
    pos_testing_path = 'face.test/test/face'
    neg_testing_path = 'face.test/test/non-face'

    num_classifiers = 2
    # For performance reasons restricting feature size
    min_feature_height = 8
    max_feature_height = 10
    min_feature_width = 8
    max_feature_width = 10

    print('Loading faces..')
    faces_training = utils.load_images(pos_training_path)
    faces_ii_training = list(map(ii.to_integral_image, faces_training))
    print('..done. ' + str(len(faces_training)) + ' faces loaded.\n\nLoading non faces..')
    non_faces_training = utils.load_images(neg_training_path)
    non_faces_ii_training = list(map(ii.to_integral_image, non_faces_training))
    print('..done. ' + str(len(non_faces_training)) + ' non faces loaded.\n')

    # classifiers are haar like features
    classifiers = ab.learn(faces_ii_training, non_faces_ii_training, num_classifiers, min_feature_height, max_feature_height, min_feature_width, max_feature_width)
    # Provide a random image path or specify an image for testing
    random_image_path = 'Image/low_diff.jpg'  # Change this to the path of your image

    # Test the image using the classifier
    detected_faces = test_random_image(classifiers, random_image_path)
    print(f"Number of faces detected in {random_image_path}: {detected_faces}")

if __name__ == "__main__":
    main()
