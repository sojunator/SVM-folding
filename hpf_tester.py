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


print(read_data_from_folder("datasets"))

X, y = make_blobs(n_samples=40,
                  n_features=2,
                  centers=2,
                  random_state=6)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

test = HPF(X_train, y_train)

# Ideally, resulst should be the same, but they ain't due to overotation
print("\nOrginal classifier")
print(test.classify(X_test, False) - y_test)

print("\nhpf classifier")
print(test.classify(X_test) - y_test)
