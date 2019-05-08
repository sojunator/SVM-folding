from hpf import HPF
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler,  Normalizer
from sklearn.datasets import make_blobs, load_breast_cancer
from hpf import HPF
from old_hpf import old_HPF
import datetime

result_dict = {}
time_dict = {}

def normalize_data(data_points):

    # Perform linear search in each dim for min and max value
    max_values = np.amax(data_points, 0)
    min_values = np.amin(data_points, 0)

    data_points = data_points - min_values
    data_points = data_points / max_values


    return data_points

def plot_hpf(hpf, ax, XX, YY, colour='k'):
    """
    Plots a clf, with margins, colour will be black
    """
    if len(hpf.rotation_data) > 0:
        intersection_point, primary_support_vector, left_or_right, (right_clf, left_clf) = hpf.rotation_data[-1]
        clf = hpf.clf
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        w = clf.coef_[0]

        k = w[1] / w[0]
        x = np.linspace(0,0.5,2)
        m = primary_support_vector[1] - k * primary_support_vector[0]

        plt.plot(x, k*x+m, '-r', label='Splitting plane')
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors=colour, label="Support vectors")

        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors=colour)

        ax.scatter(hpf.primary_support_vector[0], hpf.primary_support_vector[1], s=100,
               linewidth=2, facecolors='none', edgecolors='b', label="Primary support vector")


        Z = right_clf.decision_function(xy).reshape(XX.shape)

        ax.contour(XX, YY, Z, colors='m', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

        Z = left_clf.decision_function(xy).reshape(XX.shape)

        ax.contour(XX, YY, Z, colors='y', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

        Z = hpf.old_clf.decision_function(xy).reshape(XX.shape)

        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def plot(hpf):
    """
    God function that removes all the jitter from main
    """
    X = np.array(hpf.data_points)
    if (len(X[0]) > 2):
     X = [x[:2] for x in X]
    plt.scatter(X[:, 0], X[:, 1], c=hpf.data_labels, s=30, cmap=plt.cm.Paired)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    #plt.axvline(x=splitting_point, color='k')

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)

    YY, XX = np.meshgrid(yy, xx)


    plot_hpf(hpf, ax, XX, YY, 'g')



    plt.legend(loc='upper left')
    #plt.show()

def read_data_from_folder(folder_name):
    onlyfiles = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
    datasets = {}

    for file in onlyfiles:
            df = pd.read_csv(folder_name + "/" + file)
            temp = []
            for row in df.iterrows():
                index, data = row
                temp.append(data.tolist())

            data_points = np.array([row[:-1] for row in temp])
            data_points = data_points.astype('float64')
            data_labels = np.array([row[-1] for row in temp])
            datasets[file] = (data_points, data_labels)


    return datasets

def plot_3d(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    n = 100

    x1 = []
    y1 = []
    z1 = []
    x2 = []
    y2 = []
    z2 = []

    for index, label in enumerate(data[1]):
        #print(index)
        if label == 0:

            x1.append(data[0][index][0])
            y1.append(data[0][index][1])
            z1.append(data[0][index][2])

        elif label == 1:
            x2.append(data[0][index][0])
            y2.append(data[0][index][1])
            z2.append(data[0][index][2])

    ax.scatter(x1, y1, z1, c='r', marker='o')
    ax.scatter(x2, y2, z2, c='b', marker='^')


    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')



def plot_clf(clf, data):


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

def clean_data(training_data, c=50):
    """
    training data with labels
    return linearly separable data
    """

    clf = svm.SVC(kernel='linear', C=c)

    clf.fit(training_data[0], training_data[1])


    new_labels = clf.predict(training_data[0])


    indexes = []
    for idx, label in enumerate(new_labels):
        if not label == training_data[1][idx]:
            indexes.append(idx)

    #plot_3d(training_data)

    training_data[0] = np.delete(training_data[0], indexes, 0)
    training_data[1] = np.delete(training_data[1], indexes, 0)

    #plot_3d(training_data)
    #plt.show()

    return training_data

def write_data_to_file(result_dict, filehandle):

    for fold in result_dict:
        for classifier in result_dict[fold]:

            margin = result_dict[fold][classifier]["margin"]
            spec = result_dict[fold][classifier]["spe"]
            sen = result_dict[fold][classifier]["sen"]
            filehandle.write("{} {}\t".format(fold, margin))
            filehandle.write("{} {}\t".format(fold, sen))
            filehandle.write("{} {}\t".format(fold, spec))

            filehandle.write("\n")

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

def evaluate(classifier_str, model_ans, real_ans, i, print_ans = True):
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


    result_dict[i][classifier_str]["TP"].append(true_positives)
    result_dict[i][classifier_str]["TN"].append(true_negatives)
    result_dict[i][classifier_str]["FP"].append(false_positives)
    result_dict[i][classifier_str]["FN"].append(false_negatives)

    #true_positives, true_negatives, false_positives, false_negatives

    accuracy = (true_positives + true_negatives) / len(real_ans)
    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)

    return accuracy, sensitivity, specificity, result_dict

def write_timedict_to_file(time_dict, filehandle):
    for classifier in time_dict:
        filehandle.write("\n\n{}\n".format(classifier))
        for operation in time_dict[classifier]:
            operation_avg = time_dict[classifier][operation]
            operation_avg = sum(operation_avg) / len(operation_avg)
            filehandle.write("{} - {} avg in ms\n".format(operation, operation_avg))

        filehandle.write("\n")


def test_dataset(data_points, data_labels, name, nr_of_folds = 1):
    #test algorithms using k-fold



    K = 10

    skf = StratifiedKFold(n_splits=K)

    result_dict[0] = {}

    time_dict["SVM"] = {}
    time_dict["HPF"] = {}
    time_dict["RBF"] = {}


    for i in range(nr_of_folds):
        result_dict[i] = {}

        result_dict[i]["RBF"] = {}
        result_dict[i]["RBF"]["margin"] = 0
        result_dict[i]["RBF"]["acc"] = 0
        result_dict[i]["RBF"]["spe"] = 0
        result_dict[i]["RBF"]["sen"] = 0
        result_dict[i]["RBF"]["TP"] = []
        result_dict[i]["RBF"]["TN"] = []
        result_dict[i]["RBF"]["FP"] = []
        result_dict[i]["RBF"]["FN"] = []

        result_dict[i]["HPF"] = {}
        result_dict[i]["HPF"]["margin"] = 0
        result_dict[i]["HPF"]["acc"] = 0
        result_dict[i]["HPF"]["spe"] = 0
        result_dict[i]["HPF"]["sen"] = 0
        result_dict[i]["HPF"]["TP"] = []
        result_dict[i]["HPF"]["TN"] = []
        result_dict[i]["HPF"]["FP"] = []
        result_dict[i]["HPF"]["FN"] = []

    result_dict[0]["SVM"] = {}
    result_dict[0]["SVM"]["margin"] = 0
    result_dict[0]["SVM"]["acc"] = 0
    result_dict[0]["SVM"]["spe"] = 0
    result_dict[0]["SVM"]["sen"] = 0
    result_dict[0]["SVM"]["TP"] = []
    result_dict[0]["SVM"]["TN"] = []
    result_dict[0]["SVM"]["FP"] = []
    result_dict[0]["SVM"]["FN"] = []

    time_dict["HPF"] = {}
    time_dict["SVM"] = {}
    time_dict["RBF"] = {}

    time_dict["HPF"]["classify"] = []

    time_dict["SVM"]["classify"] = []

    time_dict["RBF"]["classify"] = []

    time_dict["HPF"]["fit"] = []

    time_dict["SVM"]["fit"] = []

    time_dict["RBF"]["fit"] = []

    avg_accuracy_svm = 0
    avg_sensitivety_svm = 0
    avg_specificity_svm = 0
    avg_svm_margin = 0
    for i in range(nr_of_folds):
        rbf = HPF(max_nr_of_folds = (i + 1), verbose = False)
        hpf = old_HPF(max_nr_of_folds = (i + 1), verbose = False) #classifier that use old hpfimplementation without rubberband folding

        #declare metrics
        avg_accuracy_hpf = 0
        avg_sensitivety_hpf = 0
        avg_specificity_hpf = 0

        avg_accuracy_rbf = 0
        avg_sensitivety_rbf = 0
        avg_specificity_rbf = 0


        avg_hpf_margin = 0
        avg_rbf_margin = 0

        avg_org_margin = 0

        index = 0

        for train_index, test_index in skf.split(data_points, data_labels): # runs k-tests

            print("K fold k =", index)
            index = index + 1
            X_train, X_test = data_points[train_index], data_points[test_index] #split data into one trainging part and one test part
            Y_train, Y_test = data_labels[train_index], data_labels[test_index] # do the same with the labels

            X_train, Y_train = clean_data([X_train, Y_train]) #Clean the training data, but not the test data


            # Fit RBF
            rbf_start_time = datetime.datetime.now()
            rbf_old_margin, rbf_new_margin = rbf.fit(X_train, Y_train, time_dict) #train
            rbf_fit_time = datetime.datetime.now() - rbf_start_time
            avg_rbf_margin += rbf_new_margin

            time_dict["RBF"]["fit"].append(rbf_fit_time.total_seconds()*1000)

            # Fit HPF
            hpf_start_time = datetime.datetime.now()
            hpf_old_margin, hpf_new_margin = hpf.fit(X_train, Y_train) #train
            hpf_fit_time = datetime.datetime.now() - hpf_start_time
            avg_hpf_margin += hpf_new_margin

            time_dict["HPF"]["fit"].append(hpf_fit_time.total_seconds()*1000)

            # Classify HPF
            hpf_start_time = datetime.datetime.now()
            hpf_ans = hpf.classify(X_test) #old hpf
            hpf_classify_time = datetime.datetime.now() - hpf_start_time
            time_dict["HPF"]["classify"].append(hpf_classify_time.total_seconds()*1000)

            # Classify RBF
            rbf_start_time = datetime.datetime.now()
            rbf_ans = rbf.classify(X_test) #new hpf
            rbf_classify_time = datetime.datetime.now() - rbf_start_time
            time_dict["RBF"]["classify"].append(rbf_classify_time.total_seconds()*1000)


            # Classify SVM
            # Classift does not improve over folds
            if i == 0:
                svm_start_time = datetime.datetime.now()
                svm_ans = rbf.classify(X_test, False) # state of the art svm
                svm_classify_time = datetime.datetime.now() - svm_start_time
                time_dict["SVM"]["classify"].append(svm_classify_time.total_seconds()*1000)

                acc, sen, spe, result_svm_tmp = evaluate("SVM", svm_ans, Y_test, i)
                avg_accuracy_svm += acc
                avg_sensitivety_svm += sen
                avg_specificity_svm += spe
                avg_svm_margin += rbf_old_margin


            #compare with expected labels
            acc, sen, spe, result_hpf_tmp = evaluate("HPF", hpf_ans, Y_test, i)
            avg_accuracy_hpf += acc
            avg_sensitivety_hpf += sen
            avg_specificity_hpf += spe

            acc, sen, spe, result_rbf_tmp = evaluate("RBF", rbf_ans, Y_test, i)
            avg_accuracy_rbf += acc
            avg_sensitivety_rbf += sen
            avg_specificity_rbf += spe


        if i == 0:
            result_dict[i]["SVM"]["margin"] = avg_svm_margin / K
            result_dict[i]["SVM"]["acc"] = avg_accuracy_svm / K
            result_dict[i]["SVM"]["spe"] = avg_specificity_svm / K
            result_dict[i]["SVM"]["sen"] = avg_sensitivety_svm  / K


        result_dict[i]["HPF"]["margin"] = avg_hpf_margin / K
        result_dict[i]["HPF"]["acc"] = avg_accuracy_hpf / K
        result_dict[i]["HPF"]["spe"] = avg_specificity_hpf / K
        result_dict[i]["HPF"]["sen"] = avg_sensitivety_hpf  / K

        result_dict[i]["RBF"]["margin"] = avg_rbf_margin / K
        result_dict[i]["RBF"]["acc"] = avg_accuracy_rbf / K
        result_dict[i]["RBF"]["spe"] = avg_specificity_rbf / K
        result_dict[i]["RBF"]["sen"] = avg_sensitivety_rbf  / K

    file = open(name + ".data", "w+")
    write_data_to_file(result_dict, file)
    file.close()
