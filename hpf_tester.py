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

X, y = make_blobs(n_samples=75,
                  n_features=2,
                  centers=2,
                  random_state=6)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=42)

hpf = HPF(max_nr_of_folds=100, verbose=True)


hpf.fit(X_train, Y_train)


plot(hpf)

print("\nHPF GIVEN ANSWER: ", hpf.classify(X_test))
print("\nEXPECTED  ANSWER: ", Y_test)
