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
from hpf_helpers import plot, read_data_from_folder, clean_data, plot_3d
from hpf import HPF


def plot_clf(data):


    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for index, label in enumerate(data[1]):
        #print(index)
        if label == 0:

            x1.append(data[0][index][0])
            y1.append(data[0][index][1])

        elif label == 1:
            x2.append(data[0][index][0])
            y2.append(data[0][index][1])


    plt.plot(x1, y1, 'ro')
    plt.plot(x2, y2, 'go')
    plt.show()

def evaluate(model_ans, real_ans, write_to_file = False, print_ans = True):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for compare in zip(model_ans, real_ans):
        if compare[0] == compare[1]: #true
            if compare[0] == 1:#positive
                true_positives += 1
            elif compare[0] == 0:#negative
                true_negatives += 1
        elif (compare[0] != compare[1]):#false
            if compare[0] == 1:#positive
                false_positives += 1
            elif compare[0] == 0:#negative
                false_negatives += 1

    if print_ans:
        print("True positives:", true_positives)
        print("True negatives:", true_negatives)
        print("False positives:", false_positives)
        print("False negatives:", false_negatives)

    return true_positives, true_negatives, false_positives, false_negatives 


#Exception when divide when zero
np.seterr(all='warn')
warnings.filterwarnings('error')

data_set = read_data_from_folder("datasets")

X_test = np.array([[-3,-4,0,0,0,0,0]])
Y_test = np.array([1])

X_train, Y_train = data_set["bmi.csv"]



#X_train, Y_train = data_set["hpf_test3D.csv"]
X_train, Y_train = data_set["dimred.csv"]
X_train, Y_train = data_set["liver.csv"]

#X_train, Y_train = make_blobs(n_samples=40,n_features=2,centers=2,random_state=6)


X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)


print(len(Y_train))
X_train, Y_train = clean_data([X_train, Y_train], 100)
print(len(Y_train))
hpf = HPF(max_nr_of_folds=100, verbose=False)

hpf.fit(X_train, Y_train)

hpf_ans = hpf.classify(X_test)
svm_ans = hpf.classify(X_test, False)

#print("\nNo rotation GIVEN ANSWER: ", hpf.classify(X_test, False))


evaluate(hpf_ans, Y_train)
evaluate(svm_ans, Y_train)
