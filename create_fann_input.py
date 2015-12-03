import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
import os

mnist = fetch_mldata("MNIST Original")
X_train, y_train = mnist.data / 255., mnist.target
#print X_train.shape

pca = PCA(n_components=50)
X_transformed = pca.fit_transform(X_train)

#create the y_vector
n = 10 #10 labels 1 to 10 both inclusive
y_vector = [['1' if i == j else '0' for i in range(n)] for j in range(n)]
#print y_vector
#print X_transformed.shape

#print X_transformed[0].tolist()

#start writing to stdout in the format of fann input file
#The first line consists of three numbers: The first is the number of training pairs in the file, the second is the number of inputs and the third is the number of outputs. The rest of the file is the actual training data, consisting of one line with inputs, one with outputs etc.

print X_transformed.shape[0], len(X_transformed[0]),10
for i, label in enumerate(y_train):
  print ' '.join(list(map(str,X_transformed[i].tolist())))
  print ' '.join(y_vector[int(label)])
    
