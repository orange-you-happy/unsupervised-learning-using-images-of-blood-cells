import numpy as np
import pandas as pd
from preprocess import load_data
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from config import SIZE, PCA_N_COMPONENT
from evaluate_accuracy import evaluate_accuracy

if __name__ == "__main__":
    X_train, X_test, y_test = load_data(SIZE)
    
    #Input takes in a 2d matrix. We will squish down the image, so each row of the matrix
    #contains all pixels of an image. 
    #We divide each value by 255.0 to map values to between 0 and 1.
    X_train = X_train.reshape(len(X_train), -1) / 255.0
    X_test = X_test.reshape(len(X_test), -1) / 255.0

    #Each row is SIZE x SIZE x 3 long, so we use PCA to reduce the length without losing detail:
    pca = PCA(n_components = PCA_N_COMPONENT, random_state = 0)
    
    #Fit_transform trains on the data before transforming it, transform just transforms it
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    k = len(np.unique(y_test)) #Number of clusters we group them up to
    kMeans = KMeans(n_clusters = k, n_init = 10, random_state = 0)
    kMeans.fit(X_train)
    
    clusters = kMeans.predict(X_test)
    evaluate_accuracy(y_test, clusters)
    
    