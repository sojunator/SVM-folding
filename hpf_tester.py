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


X_train = data_points = np.array([[0.0, 48.0], [2.0, 40.0], [1.0, 37.0], [0.0, 21.0], [2.0, 25.0], [6.0, 22.0], [5.0, 17.0], [6.0, 19.0], [5.0, 42.0], [6.0, 52.0]])
Y_train = data_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 0])

X_test = np.array([[6,45], [6, 20]])
Y_test = np.array([0,1])

hpf = HPF(verbose=True)

hpf.fit(X_train, Y_train)

print("\nHPF GIVEN ANSWER: ", hpf.classify(X_test))
print("\nEXPECTED  ANSWER: ", Y_test)

plot(hpf)
