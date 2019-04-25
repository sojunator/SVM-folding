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

X_test = np.array([[-3,-4,0,0,0,0,0]])
Y_test = np.array([1])



#X_train, Y_train = data_set["hpf_test3D.csv"]
X_train, Y_train = data_set["dimred.csv"]
X_train, Y_train = data_set["bmi.csv"]

#X_train, Y_train = make_blobs(n_samples=40,n_features=2,centers=2,random_state=6)


X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.75, random_state=42)

hpf = HPF(max_nr_of_folds=100, verbose=False)

hpf.fit(X_train, Y_train)


print("\nHPF GIVEN ANSWER: ", hpf.classify(X_test))

#print("\nNo rotation GIVEN ANSWER: ", hpf.classify(X_test, False))
print("\nEXPECTED  ANSWER: ", Y_test)
