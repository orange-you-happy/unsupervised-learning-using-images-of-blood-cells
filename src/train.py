import numpy as np
from preprocess import load_data
from model import gradient_descent, forward_prop, get_predictions, get_accuracy
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    np.random.seed(0)
    X_train, X_test, y_test = load_data()
    X_train, X_test, y_test = X_train.T, X_test.T, y_test.T
    print("Training:")
