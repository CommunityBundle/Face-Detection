import cv2

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, or replace with the camera index

while True:
    # Capture each frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale (Haar cascade works on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw bounding boxes around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the frame with detections
    cv2.imshow("Face Detection", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

# An effort to implement the Viola-Jones algorithm from scratch in Python using OpenCV and scikit-learn.
import matplotlib.pyplot as plt
from PIL import Image

# Open the PGM file
image = Image.open('face.test/test/face/cmu_0000.pgm')

# Display the image
plt.imshow(image, cmap='gray')
plt.axis('off')  # Turn off axis
plt.show()

# import numpy as np
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# import cv2
# import os
# import glob

# # Define feature types
# feature_types = ['two_horizontal', 'two_vertical', 'three_horizontal', 'four_square']

# # Step 1: Load and preprocess images
# def load_images(folder_path, size=(24, 24)):
#     images = []
#     for img_path in glob.glob(folder_path + '/*.pgm'):
#         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         if img is not None:
#             img = cv2.resize(img, size)
#             images.append(img)
#     return np.array(images)

# # Step 2: Compute integral image
# def compute_integral_image(image):
#     return image.cumsum(axis=0).cumsum(axis=1).astype(np.int64)

# # Step 3: Haar-like features
# def calculate_haar_features(integral_image, x, y, w, h, feature_type):
#     height, width = integral_image.shape
#     if y + h >= height or x + w >= width:
#         return 0  # Return 0 if the window exceeds the image boundaries

#     if feature_type == 'two_vertical':
#         left = integral_image[y + h // 2, x + w] - integral_image[y, x + w]
#         right = integral_image[y + h, x + w] - integral_image[y + h // 2, x + w]
#         return left - right
#     elif feature_type == 'two_horizontal':
#         top = integral_image[y + h, x + w // 2] - integral_image[y + h, x]
#         bottom = integral_image[y + h, x + w] - integral_image[y + h, x + w // 2]
#         return top - bottom
#     elif feature_type == 'three_horizontal':
#         top = integral_image[y + h, x + w // 3] - integral_image[y + h, x]
#         middle = integral_image[y + h, x + 2 * w // 3] - integral_image[y + h, x + w // 3]
#         bottom = integral_image[y + h, x + w] - integral_image[y + h, x + 2 * w // 3]
#         return top - 2 * middle + bottom
#     elif feature_type == 'three_vertical':
#         left = integral_image[y + h // 3, x + w] - integral_image[y, x + w]
#         middle = integral_image[y + 2 * h // 3, x + w] - integral_image[y + h // 3, x + w]
#         right = integral_image[y + h, x + w] - integral_image[y + 2 * h // 3, x + w]
#         return left - 2 * middle + right
#     elif feature_type == 'four_square':
#         top_left = integral_image[y + h // 2, x + w // 2] - integral_image[y, x + w // 2] - integral_image[y + h // 2, x] + integral_image[y, x]
#         top_right = integral_image[y + h // 2, x + w] - integral_image[y, x + w] - integral_image[y + h // 2, x + w // 2] + integral_image[y, x + w // 2]
#         bottom_left = integral_image[y + h, x + w // 2] - integral_image[y + h // 2, x + w // 2] - integral_image[y + h, x] + integral_image[y + h // 2, x]
#         bottom_right = integral_image[y + h, x + w] - integral_image[y + h // 2, x + w] - integral_image[y + h, x + w // 2] + integral_image[y + h // 2, x + w // 2]
#         return (top_left + bottom_right) - (top_right + bottom_left)
#     return 0

# # Step 4: Extract features
# def extract_features(images, feature_types):
#     features = []
#     for img in images:
#         integral_img = compute_integral_image(img)
#         img_features = []
#         for feature_type in feature_types:
#             img_features.append(calculate_haar_features(integral_img, 0, 0, 24, 24, feature_type))
#         features.append(img_features)
#     return np.array(features)

# # Step 5: Train classifier
# def train_adaboost(features, labels, n_estimators=200):
#     clf = AdaBoostClassifier(
#         estimator=DecisionTreeClassifier(max_depth=1),
#         n_estimators=n_estimators,
#         algorithm='SAMME'
#     )
#     clf.fit(features, labels)
#     return clf

# # Step 6: Save classifier
# def save_classifier(classifier, filename):
#     import joblib
#     joblib.dump(classifier, filename)

# # Step 7: Load classifier
# def load_classifier(filename):
#     import joblib
#     return joblib.load(filename)

# # Step 8: Sliding window
# def sliding_window(image, step_size, window_size):
#     for y in range(0, image.shape[0] - window_size[1], step_size):
#         for x in range(0, image.shape[1] - window_size[0], step_size):
#             yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

# # Step 9: Non-Maximum Suppression
# def non_max_suppression(boxes, overlapThresh=0.3):
#     if len(boxes) == 0:
#         return []

#     boxes = np.array(boxes)  # Convert list to NumPy array
#     if boxes.dtype.kind == "i":
#         boxes = boxes.astype("float")

#     pick = []
#     x1 = boxes[:, 0]
#     y1 = boxes[:, 1]
#     x2 = boxes[:, 0] + boxes[:, 2]
#     y2 = boxes[:, 1] + boxes[:, 3]

#     area = (x2 - x1 + 1) * (y2 - y1 + 1)
#     idxs = np.argsort(y2)

#     while len(idxs) > 0:
#         last = len(idxs) - 1
#         i = idxs[last]
#         pick.append(i)

#         xx1 = np.maximum(x1[i], x1[idxs[:last]])
#         yy1 = np.maximum(y1[i], y1[idxs[:last]])
#         xx2 = np.minimum(x2[i], x2[idxs[:last]])
#         yy2 = np.minimum(y2[i], y2[idxs[:last]])

#         w = np.maximum(0, xx2 - xx1 + 1)
#         h = np.maximum(0, yy2 - yy1 + 1)

#         overlap = (w * h) / area[idxs[:last]]

#         idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

#     return boxes[pick].astype("int")

# # Step 10: Face Detection
# def detect_faces(image, clf, feature_types, confidence_threshold=0.5):
#     detections = []
#     integral_image = compute_integral_image(image)
#     window_size = (24, 24)
#     step_size = 24
#     for (x, y, window) in sliding_window(image, step_size, window_size):
#         if window.shape[:2] == window_size:
#             features = extract_features([window], feature_types)
#             prediction = clf.predict_proba(features)[0][1]
#             if prediction >= confidence_threshold:
#                 detections.append((x, y, window_size[0], window_size[1]))
            
#             # Visualization of the sliding window
#             clone = image.copy()
#             cv2.rectangle(clone, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 2)
#             cv2.imshow("Sliding Window", clone)
#             cv2.waitKey(30)  # Adjust delay for better visualization

#     cv2.destroyAllWindows()
#     return non_max_suppression(detections)

# # Main Simulation
# if __name__ == "__main__":
#     classifier_file = "face_detector.pkl"
    
#     # Check if classifier already exists
#     if os.path.exists(classifier_file):
#         print("Loading existing classifier...")
#         classifier = load_classifier(classifier_file)
#     else:
#         print("Training classifier...")
#         pos_samples = load_images('face.train/train/face')
#         neg_samples = load_images('face.train/train/non-face')
        
#         pos_samples = [cv2.resize(img, (24, 24)) for img in pos_samples if img is not None]
#         neg_samples = [cv2.resize(img, (24, 24)) for img in neg_samples if img is not None]
        
#         pos_features = []
#         neg_features = []
        
#         for img in pos_samples:
#             integral = compute_integral_image(img)
#             img_features = [calculate_haar_features(integral, 0, 0, 24, 24, ft) for ft in feature_types]
#             pos_features.append(img_features)
        
#         for img in neg_samples:
#             integral = compute_integral_image(img)
#             img_features = [calculate_haar_features(integral, 0, 0, 24, 24, ft) for ft in feature_types]
#             neg_features.append(img_features)
        
#         X = np.vstack((pos_features, neg_features))
#         y = np.hstack((np.ones(len(pos_features)), np.zeros(len(neg_features))))
        
#         classifier = train_adaboost(X, y)
        
#         # Save the classifier for future use
#         save_classifier(classifier, classifier_file)
    
#     # Test on a new image
#     test_image = cv2.imread("Image/low_diff.jpg", cv2.IMREAD_GRAYSCALE)
#     detections = detect_faces(test_image, classifier, feature_types)
#     print(f"Detected {len(detections)} faces")
#     for (x, y, w, h) in detections:
#         cv2.rectangle(test_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
#     cv2.imshow("Detections", test_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()