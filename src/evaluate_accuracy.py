import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, confusion_matrix
from config import LABEL_NAMES

#Clustering assigns each image a group number, but these numbers are arbitrary and won't match the labels
#(cluster 5 might be label 2), so we need to match each cluster to the label it overlaps with most before scoring.


def match_clusters(y_true, clusters):
    #To find which cluster represents which label, we count the number of images in each label and 
    #clusters.
    table = pd.crosstab(y_true, clusters)

    #Linear sum assignment then compares the numbers, matches the closest ones, finding which cluster represents which label.
    rows, cols = linear_sum_assignment(table.values, maximize = True)
    mapping = {table.columns[c]: table.index[r] for r, c in zip(rows, cols)}

    #Turn each cluster into its correct label, then score
    predicted_labels = np.array([mapping[c] for c in clusters])
    return predicted_labels

def evaluate_accuracy(y_true, clusters):
    predicted_labels = match_clusters(y_true, clusters)
    labels = np.unique(y_true)
    confusion = confusion_matrix(y_true, predicted_labels, labels = labels)
    print("Row: Actual cell type, Column: Predicted cell type")
    names = [LABEL_NAMES[label] for label in labels]
    print(pd.DataFrame(confusion, index = names, columns = names).to_string())
    accuracy = accuracy_score(y_true, predicted_labels)
    print(f"Accuracy: {accuracy:.2%}")
    return accuracy