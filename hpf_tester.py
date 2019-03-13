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


data_set = read_data_from_folder("datasets")
#bmi_data_points, bmi_data_labels = data_set["bmi.csv"]
#X_train, X_test, Y_train, Y_test = train_test_split(bmi_data_points, bmi_data_labels, test_size =0.33, random_state=42)
#X_train = data_points = np.array([[0.0, 48.0], [2.0, 40.0], [1.0, 37.0], [0.0, 21.0], [2.0, 25.0], [6.0, 22.0], [5.0, 17.0], [6.0, 19.0], [5.0, 42.0], [6.0, 52.0]])
#Y_train = data_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 0])

X_train, Y_train = data_set["align_test_4D.csv"]

X_test = np.array([[-24.4,22.45, -26.4], [22.2, 21.4, 23.3], [-1.3, -21.1, 4.4]])
Y_test = np.array([1,1,0])

#X_train, X_test, Y_train, Y_test = train_test_split(data_points, data_labels, test_size =0.33, random_state=42)

hpf = HPF()

hpf.fit(X_train, Y_train)


print("\nHPF GIVEN ANSWER: ", hpf.classify(X_test))
print("\nEXPECTED  ANSWER: ", Y_test)

plot(hpf)
