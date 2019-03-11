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


#print(read_data_from_folder("datasets"))

#X, y = make_blobs(n_samples=40, n_features=2, centers=2, random_state=6)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#test = HPF(X_train, y_train)
data_points = np.array([[2.,7.,0.,2.,2.],[1.,5.,1.,2.,2.], [-1.,5.,2.,2.,2.], [-2.,3.,4.,5.,2.], [1.,2.,4.,5.,2.], [0.,0.,5.,5.,2.], [0.,4.,-1.,2.,2.], [0.,1.,5.,5.,2.]])
#data_labels = np.array([1,1,1,0,0,0,1,0])

data_set = read_data_from_folder("datasets")
bmi_data_points, bmi_data_labels = data_set["bmi.csv"]

X_train, X_test, Y_train, Y_test = train_test_split(bmi_data_points, bmi_data_labels, test_size =0.33, random_state=42)

hpf = HPF()

hpf.fit(X_train, Y_train)

print("\nHPF GIVEN ANSWER: ", hpf.classify(X_test)) #[0.,2.,6.,5.,2.]
print("\nEXPECTED  ANSWER: ", Y_test)


# Ideally, resulst should be the same, but they ain't due to overotation
#print("\nOrginal classifier")
#print(test.classify(X_test, False) - y_test)

#print("\nhpf classifier")
#print(test.classify(X_test) - y_test)
