import numpy as np
from config import N_HIDDEN, ALPHA, PCA_N_COMPONENT

def init_params():
    W1 = np.random.rand(N_HIDDEN, PCA_N_COMPONENT) - 0.5
    W2 = np.random.rand(PCA_N_COMPONENT, N_HIDDEN) - 0.5

    b1 = np.zeros((N_HIDDEN, 1))
    b2 = np.zeros((PCA_N_COMPONENT, 1))
    
    return W1, W2, b1, b2

def tanh(Z):
    return np.tanh(Z)

def forward_prop(W1, W2, b1, b2, X):
    Z1 = W1 @ X + b1
    #A1 labels: Each row is a neuron, each column is an image
    A1 = tanh(Z1)
    
    
    Z2 = W2 @ A1 + b2
    #A2 labels: Each row is a pixel, each column is an image
    A2 = Z2 #Unsupervised learning, no activation function for this
    
    return Z1, Z2, A1, A2

def backward_prop(Z1, A1, A2, W2, X, Y): #Find how much the parameters should change after prediction
    pass

def update_params():
    pass

def get_predictions():
    pass

def gradient_descent(): #Actually updates the parameters
    pass





    


