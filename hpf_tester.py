from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import math
import json

from hpf_helpers import plot, read_data_from_folder
from hpf import HPF

#X, y = make_blobs(n_samples=40,n_features=2,centers=2,random_state=6)

data_set = read_data_from_folder("datasets")

#X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.0, random_state=42)

X_train, Y_train = data_set["hpf_test.csv"]

X_test, Y_test = np.array([[-5.0, -1.0]]), np.array([0])

hpf = HPF(max_nr_of_folds=100, verbose=False)

#hpf = HPF(lambda p, i, r : np.matmul(p - i, r) + i, 1)


hpf.fit(X_train, Y_train)

plot(hpf)

print("\nHPF GIVEN ANSWER: ", hpf.classify(X_test))
print("\nEXPECTED  ANSWER: ", Y_test)
