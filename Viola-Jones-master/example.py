import violajones.IntegralImage as ii
import violajones.AdaBoost as ab
import violajones.Utils as utils
import pickle
import progressbar
import matplotlib.pyplot as plt
from multiprocessing import Pool
def plot_metrics(precision, recall, f1_score, accuracy):
    metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    values = [precision, recall, f1_score, accuracy]

    # Plotting the metrics
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'salmon', 'orange'])
    plt.ylim(0, 1)  # Metrics are between 0 and 1
    plt.title('Evaluation Metrics', fontsize=16)
    plt.ylabel('Value', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Adding text on top of bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f'{value:.2f}', 
                 ha='center', va='bottom', fontsize=12)

    # Display the graph
    plt.show()
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

    # Save the classifier model using pickle
    with open('classifier_model.pkl', 'wb') as f:
        pickle.dump(classifiers, f)
    print('Classifier model saved as "classifier_model.pkl".')

    # Test classifiers on testing data
    print('Loading test faces..')
    faces_testing = utils.load_images(pos_testing_path)
    faces_ii_testing = list(map(ii.to_integral_image, faces_testing))
    print('..done. ' + str(len(faces_testing)) + ' faces loaded.\n\nLoading test non faces..')
    non_faces_testing = utils.load_images(neg_testing_path)
    non_faces_ii_testing = list(map(ii.to_integral_image, non_faces_testing))
    print('..done. ' + str(len(non_faces_testing)) + ' non faces loaded.\n')

    print('Testing selected classifiers..')
    correct_faces = 0
    correct_non_faces = 0
    correct_faces = sum(utils.ensemble_vote_all(faces_ii_testing, classifiers))
    correct_non_faces = len(non_faces_testing) - sum(utils.ensemble_vote_all(non_faces_ii_testing, classifiers))

    print('..done.\n\nResult:\n      Faces: ' + str(correct_faces) + '/' + str(len(faces_testing))
          + '  (' + str((float(correct_faces) / len(faces_testing)) * 100) + '%)\n  non-Faces: '
          + str(correct_non_faces) + '/' + str(len(non_faces_testing)) + '  (' 
          + str((float(correct_non_faces) / len(non_faces_testing)) * 100) + '%)')
    print('Testing selected classifiers..')
    # Predictions for faces and non-faces
    face_predictions = utils.ensemble_vote_all(faces_ii_testing, classifiers)
    non_face_predictions = utils.ensemble_vote_all(non_faces_ii_testing, classifiers)

    # Calculate metrics
    TP = sum(face_predictions)  # Correctly classified faces
    FN = len(faces_testing) - TP  # Missed faces
    TN = len(non_faces_testing) - sum(non_face_predictions)  # Correctly classified non-faces
    FP = sum(non_face_predictions)  # Non-faces classified as faces

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (len(faces_testing) + len(non_faces_testing))

    # Print results
    print('..done.\n\nResults:')
    print(f'      Faces: {TP}/{len(faces_testing)}  ({(TP / len(faces_testing)) * 100:.2f}%)')
    print(f'  Non-Faces: {TN}/{len(non_faces_testing)}  ({(TN / len(non_faces_testing)) * 100:.2f}%)')
    print(f'\nEvaluation Metrics:')
    print(f'  Precision: {precision:.4f}')
    print(f'  Recall: {recall:.4f}')
    print(f'  F1-Score: {f1_score:.4f}')
    print(f'  Accuracy: {accuracy:.4f}')
    plot_metrics(precision, recall, f1_score, accuracy)

if __name__ == "__main__":
    main()
