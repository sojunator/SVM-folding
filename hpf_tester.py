from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

from hpf import HPF

import matplotlib.pyplot as plt
import numpy as np
import math



data_points, data_labels = make_blobs(n_samples=40,
                                     n_features=2,
                                     centers=2,
                                     random_state=6)

test = HPF(data_points, data_labels)


print(test.classify(np.array([[6.4, -4.8],[5.4, -7.5]])))
