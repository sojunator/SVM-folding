from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold
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
from old_hpf import old_HPF

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

    #true_positives, true_negatives, false_positives, false_negatives 
    
    accuracy = (true_positives + true_negatives) / len(real_ans)
    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)

    return accuracy, sensitivity, specificity


#Exception when divide when zero
np.seterr(all='warn')
warnings.filterwarnings('error')

data_set = read_data_from_folder("datasets") # Load data


#X_train, Y_train = make_blobs(n_samples=40,n_features=2,centers=2,random_state=6)
#X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=30)

#X_train, Y_train = data_set["bmi.csv"]
#X_train, Y_train = data_set["hpf_test3D.csv"]
#data_points, data_labels = data_set["dimred.csv"]
data_points, data_labels = data_set["liver.csv"]


hpf = HPF(max_nr_of_folds=100, verbose=False) 
old_hpf = old_HPF() #classifier that use old hpfimplementation without rubberband folding


#test algorithms using k-fold
K = 5
sk_kf = KFold(n_splits=K, shuffle=True)

#declare metrics
avg_accuracy_old_hpf = 0
avg_sensitivety_old_hpf = 0
avg_specificity_old_hpf = 0

avg_accuracy_hpf = 0
avg_sensitivety_hpf = 0
avg_specificity_hpf = 0

avg_accuracy_svm = 0
avg_sensitivety_svm = 0
avg_specificity_svm = 0

new_margin = 0
old_margin = 0
old_new_margin = 0
old_old_margin = 0
index = 0
for train_index, test_index in sk_kf.split(data_points): # runs k-tests
    
    print("K fold k =", index)
    index = index +1
    X_train, X_test = data_points[train_index], data_points[test_index] #split data into one trainging part and one test part
    Y_train, Y_test = data_labels[train_index], data_labels[test_index] # do the same with the labels

    X_train, Y_train = clean_data([X_train, Y_train], 50) #Clean the training data, but not the test data

    print("running HPF")
    old_margin, new_margin = hpf.fit(X_train, Y_train) #train
    print("running old")
    old_old_margin, old_new_margin = old_hpf.fit(X_train, Y_train) #train

    old_hpf_ans = old_hpf.classify(X_test) #old hpf
    improved_hpf_ans = hpf.classify(X_test) #new hpf
    svm_ans = hpf.classify(X_test, False) # state of the art svm

    #compare with expected labels
    acc, sen, spe = evaluate(old_hpf_ans, Y_test) 
    avg_accuracy_old_hpf += acc
    avg_sensitivety_old_hpf += sen
    avg_specificity_old_hpf += spe

    acc, sen, spe = evaluate(improved_hpf_ans, Y_test) 
    avg_accuracy_hpf += acc
    avg_sensitivety_hpf += sen
    avg_specificity_hpf += spe
    
    acc, sen, spe = evaluate(svm_ans, Y_test)
    avg_accuracy_svm += acc
    avg_sensitivety_svm += sen
    avg_specificity_svm += spe


print("\n\nOld old Margin: ", old_old_margin)
print("New old Margin: ", old_new_margin)
print("old Margin Change: ", old_new_margin - old_old_margin)

print("\nOld Margin: ", old_margin)
print("New Margin: ", new_margin)
print("Margin Change: ", new_margin - old_margin)

print("\n\nAccuracy old :", avg_accuracy_old_hpf / K)
print("Sensitivety old :", avg_sensitivety_old_hpf / K)
print("Specificity :", avg_specificity_old_hpf / K)

print("\n\nAccuracy HPF :", avg_accuracy_hpf / K)
print("Sensitivety HPF :", avg_sensitivety_hpf / K)
print("Specificity HPF :", avg_specificity_hpf / K)

print("\n\nAccuracy SVM :", avg_accuracy_svm / K)
print("Sensitivety SVM :", avg_sensitivety_svm / K)
print("Specificity SVM :", avg_specificity_svm / K)