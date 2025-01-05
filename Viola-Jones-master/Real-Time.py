import numpy as np
import cv2
from PIL import Image
from functools import partial
import violajones.IntegralImage as ii
import violajones.Utils as utils
import pickle

# Load the classifier model from the saved file
with open('classifier_model.pkl', 'rb') as f:
    classifiers = pickle.load(f)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Non-maximum suppression function
def non_max_suppression(boxes, overlap_thresh=0.5):
    if len(boxes) == 0:
        return []

    # Convert to float to avoid integer overflow when calculating coordinates
    boxes = np.array(boxes, dtype=float)
    pick = []

    # Get the coordinates of the boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    # Compute the area of each bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort the bounding boxes by their bottom-right y-coordinate
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        # Pick the last box
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find all other boxes that overlap with the picked box
        suppress = [last]
        for pos in range(last):
            j = idxs[pos]

            # Compute the intersection area
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # Compute the intersection width and height
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # Compute the ratio of overlap
            overlap = (w * h) / area[j]

            if overlap > overlap_thresh:
                suppress.append(pos)

        # Remove suppressed boxes
        idxs = np.delete(idxs, suppress)

    # Return the picked boxes
    return [boxes[i] for i in pick]

# Real-time face detection loop
while True:
    # Read the current frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Perform sliding window detection across the image
    window_size = (160, 160)  # Increased window size for better detection
    step_size = 80  # Increased step size to reduce the number of detections

    detected_faces = []

    # Slide the window across the image
    for y in range(0, gray.shape[0] - window_size[1], step_size):
        for x in range(0, gray.shape[1] - window_size[0], step_size):
            # Extract the window from the image
            window = gray[y:y + window_size[1], x:x + window_size[0]]

            # Compute the integral image of the window
            integral_image = ii.to_integral_image(window)

            # Perform the ensemble vote for the current window
            label = utils.ensemble_vote(integral_image, classifiers)

            # If the vote is positive (i.e., face detected), record the bounding box
            if label == 1:
                detected_faces.append((x, y, window_size[0], window_size[1]))

    # Apply non-maximum suppression to remove overlapping detections
    final_faces = non_max_suppression(detected_faces, overlap_thresh=0.5)  # Increased overlap threshold

    # Draw bounding boxes for detected faces
    for (x, y, w, h) in final_faces:
        # Ensure the bounding box format is (x, y, x + w, y + h)
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

    # Print the number of detected faces
    num_faces = len(final_faces)
    print(f"Number of faces detected: {num_faces}")

    # Display the frame with detected faces
    cv2.imshow('Real-Time Face Detection', frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
