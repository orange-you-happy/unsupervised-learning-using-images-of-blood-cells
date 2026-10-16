#Maps each cell type to its label number.
LABEL_NAMES = ["monocyte", "ig", "neutrophil", "basophil", "lymphocyte", 
               "erythroblast", "eosinophil", "platelet"]

#During preprocessing, the image data is resized to SIZE x SIZE
SIZE = 64

#The PCA will reduce the length of a given row from SIZE x SIZE x 3 to PCA_N_COMPONENT
#This is to increase the speed of the code while minimising the loss in detail.
PCA_N_COMPONENT = 50

#N_HIDDEN: Number of neurons per layer, ALPHA: Learning rate, 
#ITERATIONS: Number of times the process is run
N_HIDDEN = 300
ALPHA = 1
ITERATIONS = 1000