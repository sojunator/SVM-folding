from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import math
import json
import warnings
import pdb
import os
from hpf_helpers import plot, read_data_from_folder
from hpf import HPF

#Exception when divide when zero
np.seterr(all='warn')
warnings.filterwarnings('error')

data_set = read_data_from_folder("datasets")

X_test = np.array([[1,-3,0,0,0,0,0]])
Y_test = np.array([1])


#X_train, Y_train = data_set["hpf_test3D.csv"]
X_train, Y_train = data_set["dimred.csv"]
#X_train, Y_train = data_set["bmi.csv"]

#X_train, Y_train = make_blobs(n_samples=40,n_features=2,centers=2,random_state=6)
X_test = X_train[-5:]
Y_test = Y_train[-5:]

<<<<<<< HEAD
#X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.5, random_state=42)
=======
#X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.75, random_state=42)
>>>>>>> 47a4a8f62e96db825fcbfaf7ba22e66d7828f4be
#X_test, Y_test = np.array([[80, 60, 155]]), np.array([0])
#X_test, Y_test = np.array([[0.0, 0.0, -4.0, 0.0]]), np.array([0])

hpf = HPF(max_nr_of_folds=100, verbose=False)

#hpf = HPF(lambda p, i, r : np.matmul(p - i, r) + i, 1)


hpf.fit(X_train, Y_train)

#plot(hpf)

print(" ")
print(hpf.data[0][-5:])
print("\nHPF GIVEN ANSWER: ", hpf.classify(X_test))

#print("\nNo rotation GIVEN ANSWER: ", hpf.classify(X_test, False))
print("\nEXPECTED  ANSWER: ", Y_test)
