import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str)

    args = parser.parse_args()
    file_path = args.file_path

    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            true_class, predictions = line.strip().split(':')
            true_class = int(true_class.strip())
            predictions = list(map(int, predictions.strip().strip('[]').split(',')))
            data[true_class] = predictions

    # Initialize the confusion matrix
    classes = sorted(data.keys())  # Extract all true classes
    n_classes = len(classes)
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)

    # Create a mapping from class to index for the matrix
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    # Populate the confusion matrix
    for true_class, predictions in data.items():
        predicted_class = predictions[0]  # Assume the top prediction as the predicted class
        if predicted_class in class_to_index:
            confusion_matrix[class_to_index[true_class], class_to_index[predicted_class]] += 1

    # Convert to a DataFrame for better visualization
    confusion_matrix_df = pd.DataFrame(confusion_matrix, index=classes, columns=classes)

    # Save or print the confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix_df)

    # Optionally save it to a CSV file
    confusion_matrix_df.to_csv('/netscratch/shanbhag/zero-shot-diffusion-classifier/confusion-matrices/matrix1.csv')

if __name__ == '__main__':
    main()