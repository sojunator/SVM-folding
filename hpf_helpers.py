from hpf import HPF
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler,  Normalizer
from sklearn.datasets import make_blobs, load_breast_cancer
from hpf import HPF
from old_hpf import old_HPF

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

    print("fit done")
    new_labels = clf.predict(training_data[0])


    indexes = []
    for idx, label in enumerate(new_labels):
        if not (label == training_data[1][idx]).all():
            indexes.append(idx)


    #plot_3d(training_data)

    training_data[0] = np.delete(training_data[0], indexes, 0)
    training_data[1] = np.delete(training_data[1], indexes, 0)

    #plot_3d(training_data)
    #plt.show()

    return training_data



def write_matrix_to_file(result_list, filehandle):
    for element in result_list:
        filehandle.write("K-fold nr {} \n".format(result_list.index(element)))
        for key in element:
            filehandle.write("{} : {} \n".format(key, element[key]))
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

    result_dict = {}
    result_dict["TP"] = true_positives
    result_dict["TN"] = true_negatives
    result_dict["FP"] = false_positives
    result_dict["FN"] = false_negatives

    #true_positives, true_negatives, false_positives, false_negatives

    accuracy = (true_positives + true_negatives) / len(real_ans)
    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)

    return accuracy, sensitivity, specificity, result_dict

def test_dataset(data_points, data_labels, name):
    #test algorithms using k-fold

    rbf = HPF(max_nr_of_folds=1, verbose=False)
    hpf = old_HPF(max_nr_of_folds=1, verbose=False) #classifier that use old hpfimplementation without rubberband folding

    K = 10
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


    result_hpf = []
    result_rbf = []
    result_svm = []



    for train_index, test_index in sk_kf.split(data_points): # runs k-tests

        print("K fold k =", index)
        index = index +1
        X_train, X_test = data_points[train_index], data_points[test_index] #split data into one trainging part and one test part
        Y_train, Y_test = data_labels[train_index], data_labels[test_index] # do the same with the labels

        X_train, Y_train = clean_data([X_train, Y_train], 50) #Clean the training data, but not the test data

        #print("running HPF")
        old_margin, new_margin = rbf.fit(X_train, Y_train) #train
        #print("running old")
        old_old_margin, old_new_margin = hpf.fit(X_train, Y_train) #train

        old_hpf_ans = hpf.classify(X_test) #old hpf
        improved_hpf_ans = rbf.classify(X_test) #new hpf
        svm_ans = rbf.classify(X_test, False) # state of the art svm

        #compare with expected labels
        acc, sen, spe, result_hpf_tmp = evaluate(old_hpf_ans, Y_test)
        avg_accuracy_old_hpf += acc
        avg_sensitivety_old_hpf += sen
        avg_specificity_old_hpf += spe

        acc, sen, spe, result_rbf_tmp = evaluate(improved_hpf_ans, Y_test)
        avg_accuracy_hpf += acc
        avg_sensitivety_hpf += sen
        avg_specificity_hpf += spe

        acc, sen, spe, result_svm_tmp = evaluate(svm_ans, Y_test)
        avg_accuracy_svm += acc
        avg_sensitivety_svm += sen
        avg_specificity_svm += spe

        result_svm.append(result_svm_tmp)
        result_rbf.append(result_rbf_tmp)
        result_hpf.append(result_hpf_tmp)


    file = open(name + ".data", "w+")
    file.write("\n\nHPF old Margin: {} \n".format(old_old_margin))
    file.write("HPF New Margin: {} \n".format(old_new_margin))
    file.write("HPF Margin Change: {} \n".format(old_new_margin - old_old_margin))

    file.write("\nOrginal Margin: {}  \n".format(old_margin))
    file.write("RBF Margin: {}  \n".format(new_margin))
    file.write("Margin Change: {}  \n".format(new_margin - old_margin))

    file.write("\n\nAccuracy HPF : {}  \n".format((avg_accuracy_old_hpf / K)))
    file.write("Sensitivety HPF : {}  \n".format(avg_sensitivety_old_hpf / K))
    file.write("Specificity HPF : {}  \n".format(avg_specificity_old_hpf / K))

    file.write("\n\nAccuracy RBF : {}  \n".format(avg_accuracy_hpf / K))
    file.write("Sensitivety RBF : {}  \n".format(avg_sensitivety_hpf / K))
    file.write("Specificity RBF : {}  \n".format(avg_specificity_hpf / K))

    file.write("\n\nAccuracy SVM : {}\n".format(avg_accuracy_svm / K))
    file.write("Sensitivety SVM : {} \n".format(avg_sensitivety_svm / K))
    file.write("Specificity SVM : {} \n".format(avg_specificity_svm / K))


    file.write("RBF DATA \n")
    write_matrix_to_file(result_rbf, file)
    file.write("\n\n")
    file.write("HPF DATA\n")
    write_matrix_to_file(result_hpf, file)
    file.write("\n\n")
    file.write("SVM DATA\n")
    write_matrix_to_file(result_svm, file)

    file.close()
