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
from mpl_toolkits.mplot3d import axes3d

result_dict = {}
time_dict = {}

def plot2d_from_columns(file_path, x_column_index = 0):

    ######
    #Note: plt.show() needs to be called outside the function
    #####


    data_frame = pd.read_csv(file_path)

    columns_len = len(data_frame.columns.values)
    if (x_column_index > columns_len - 1 ):
        print("Error: Columns out of range!")
        return

    data_frame.plot.line(x=data_frame.columns.values[x_column_index],y=data_frame.drop(data_frame.columns.values[x_column_index], axis=1).columns.values)




def normalize_data(data_points):

    # Perform linear search in each dim for min and max value
    max_values = np.amax(data_points, 0)
    min_values = np.amin(data_points, 0)

    data_points = data_points - min_values
    data_points = data_points / (max_values - min_values)


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


def plot_3d(data, normal = None, intercept = None):

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')


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
    ax.scatter(x2, y2, z2, c='b', marker='x')


    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)


    #point  = np.array([0.5, 0.5, 0.5])
    #normal = np.array([1, 1, 2])


    if normal is not None and intercept is not None:
        # a plane is a*x+b*y+c*z+d=0
        # [a,b,c] is the normal. Thus, we have to calculate
        # d and we're set

        # create x,y
        xx, yy = np.meshgrid([0,1], [0,1])

        #d = -point.dot(normal)

        # calculate corresponding z
        z = (-intercept -normal[0] * xx - normal[1] * yy) * 1. /normal[2]

        # plot the surface
        ax.plot_surface(xx, yy, z)




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


def sample_sphere(center,radius):

    two_pi = np.math.pi * 2.0
    point = []

    rand_radius = np.random.random_sample() *radius *2 - radius
    theta = np.random.random_sample() * two_pi
    phi = np.random.random_sample()* np.math.pi

    point.append(np.clip(rand_radius * np.math.cos(theta) * np.math.sin(phi) + center[0], 0, 1))
    point.append(np.clip(rand_radius * np.math.sin(theta) * np.math.sin(phi) + center[1], 0, 1))
    point.append(np.clip(rand_radius * np.math.cos(phi)  + center[2], 0, 1))

    return np.array([point])


def extend_data_spherical(data_points, data_labels, multiplyer = 25, radius = 1):
    if len(data_points[0]) != 3:
        print("wrong dim")

    for p in zip(data_points, data_labels):
        for i in range(0,multiplyer):
            data_points = np.append(data_points, sample_sphere(p[0],radius), axis=0)
            data_labels = np.append(data_labels, np.array(p[1]))


    return data_points, data_labels

def clean_data(training_data, c=1):
    """
    training data with labels
    return linearly separable data
    """

    clf = svm.SVC(kernel='linear', C=c)

    clf.fit(training_data[0], training_data[1])


    new_labels = clf.predict(training_data[0])

    removed_data = []
    indexes = []
    for idx, label in enumerate(new_labels):
        if not label == training_data[1][idx]:
            indexes.append(idx)

    normal = clf.coef_[0]
    clf.intercept_[0]



    #plot_3d(training_data, np.array(normal), clf.intercept_[0])

    # Copy the deleted points before removal
    for index in indexes:
        removed_data.append((training_data[0][index], training_data[1][index]))

    print(len(removed_data))
    training_data[0] = np.delete(training_data[0], indexes, 0)
    training_data[1] = np.delete(training_data[1], indexes, 0)

    #plot_3d(training_data, np.array(normal), clf.intercept_[0])

    #plt.show()

    return training_data

def dump_raw_data(result_dict, file):
    for fold in result_dict:
        for classifier in result_dict[fold]:
            file.write("{} - {}\n".format(classifier, fold))
            for key in ["TP", "TN", "FP", "FN"]:
                file.write("{} - {}\n".format(key, result_dict[fold][classifier][key]))


def write_data_to_file(result_dict, time_dict, filehandle, keys = None, special_keys = None):

    if keys == None:
        keys = ["acc", "margin", "sen", "ang"]
    if special_keys == None:
        special_keys = ["TP", "TN", "FP", "FN"]

    filehandle.write("Fold")

    for classifier in result_dict[0]:
            for entry in keys + special_keys:
                filehandle.write(",{}{}".format(classifier, entry))
    filehandle.write("\n")

    for fold in result_dict:
        filehandle.write("{}".format(fold))
        for classifier in result_dict[fold]:
            for key in keys+ special_keys:
                if key in special_keys:
                    avg_value = sum(result_dict[fold][classifier][key]) / len(result_dict[fold][classifier][key])

                    filehandle.write(",{}".format(avg_value))
                else:
                    feature = result_dict[fold][classifier][key]

                    filehandle.write(",{}".format(feature))



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


def test_dataset(data_points, data_labels, name, nr_of_folds = 1, extend = False, K = 2):
    #test algorithms using k-fold



    skf = StratifiedKFold(n_splits=K)



    time_dict["SVM"] = {}
    time_dict["HPF"] = {}
    time_dict["RBF"] = {}

    measurements = ["margin", "acc", "spe", "sen", "ang"] #graph list

    if extend:
        data_points, data_labels = extend_data_spherical(data_points, data_labels, 5, 0.3) #extend for bmi data
    for i in range(nr_of_folds + 1):
        result_dict[i] = {}
        for classifier in ["RBF", "HPF", "SVM"]:
            result_dict[i][classifier] = {}
            result_dict[i][classifier]["margin"] = 0
            result_dict[i][classifier]["acc"] = 0
            result_dict[i][classifier]["spe"] = 0
            result_dict[i][classifier]["sen"] = 0
            result_dict[i][classifier]["ang"] = 0
            result_dict[i][classifier]["TP"] = []
            result_dict[i][classifier]["TN"] = []
            result_dict[i][classifier]["FP"] = []
            result_dict[i][classifier]["FN"] = []

            time_dict[classifier][i] = {}
            time_dict[classifier][i]["classify"] = []
            time_dict[classifier][i]["fit"] = []








    index = 0
    for train_index, test_index in skf.split(data_points, data_labels): # runs k-tests

        print("K fold k =", index)
        index = index + 1
        X_train, X_test = data_points[train_index], data_points[test_index] #split data into one trainging part and one test part
        Y_train, Y_test = data_labels[train_index], data_labels[test_index] # do the same with the labels
        #X_test, Y_test = extend_data_spherical(X_test, Y_test, 20, 0.1)

        #plot_3d([X_train, Y_train])




        #plot_3d([X_train, Y_train])
        #plt.show()
        #declare metrics
        avg_accuracy_hpf = 0
        avg_sensitivety_hpf = 0
        avg_specificity_hpf = 0

        avg_accuracy_rbf = 0
        avg_sensitivety_rbf = 0
        avg_specificity_rbf = 0

        avg_accuracy_svm = 0
        avg_sensitivety_svm = 0
        avg_specificity_svm = 0
        avg_svm_margin = 0

        avg_hpf_margin = 0
        avg_rbf_margin = 0

        avg_org_margin = 0


        X_train, Y_train = clean_data([X_train, Y_train]) #Clean the training data, but not the test data
        for i in range(nr_of_folds):
            rbf = HPF(max_nr_of_folds = (i + 1), verbose = False)
            hpf = old_HPF(max_nr_of_folds = (i + 1), verbose = False) #classifier that use old hpfimplementation without rubberband folding



            # Fit RBF
            rbf_start_time = datetime.datetime.now()
            rbf_old_margin, rbf_new_margin = rbf.fit(X_train, Y_train, time_dict) #train
            rbf_fit_time = datetime.datetime.now() - rbf_start_time
            avg_rbf_margin += rbf_new_margin

            time_dict["RBF"][i]["fit"].append(rbf_fit_time.total_seconds()*1000)

            # Fit HPF
            hpf_start_time = datetime.datetime.now()
            hpf_old_margin, hpf_new_margin = hpf.fit(X_train, Y_train) #train
            hpf_fit_time = datetime.datetime.now() - hpf_start_time
            avg_hpf_margin += hpf_new_margin

            time_dict["HPF"][i]["fit"].append(hpf_fit_time.total_seconds()*1000)

            # Classify HPF
            hpf_start_time = datetime.datetime.now()
            hpf_ans = hpf.classify(X_test) #old hpf
            hpf_classify_time = datetime.datetime.now() - hpf_start_time
            time_dict["HPF"][i]["classify"].append(hpf_classify_time.total_seconds()*1000)

            # Classify RBF
            rbf_start_time = datetime.datetime.now()
            rbf_ans = rbf.classify(X_test) #new hpf
            rbf_classify_time = datetime.datetime.now() - rbf_start_time
            time_dict["RBF"][i]["classify"].append(rbf_classify_time.total_seconds()*1000)


            # Classify SVM
            # Classift does not improve over folds

            svm_start_time = datetime.datetime.now()
            svm_ans = rbf.classify(X_test, False) # state of the art svm
            svm_classify_time = datetime.datetime.now() - svm_start_time
            time_dict["SVM"][i]["classify"].append(svm_classify_time.total_seconds()*1000)

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



            result_dict[i + 1]["SVM"]["margin"] += avg_svm_margin
            result_dict[i + 1]["SVM"]["acc"] += avg_accuracy_svm
            result_dict[i + 1]["SVM"]["spe"] += avg_specificity_svm
            result_dict[i + 1]["SVM"]["sen"] += avg_sensitivety_svm


            result_dict[i + 1]["HPF"]["margin"] += avg_hpf_margin
            result_dict[i + 1]["HPF"]["acc"] += avg_accuracy_hpf
            result_dict[i + 1]["HPF"]["spe"] += avg_specificity_hpf
            result_dict[i + 1]["HPF"]["sen"] += avg_sensitivety_hpf
            result_dict[i + 1]["HPF"]["ang"] += hpf.rotation_data[i][-1]

            result_dict[i + 1]["RBF"]["ang"] += rbf.rotation_data[i][-1]
            result_dict[i + 1]["RBF"]["margin"] += avg_rbf_margin
            result_dict[i + 1]["RBF"]["acc"] += avg_accuracy_rbf
            result_dict[i + 1]["RBF"]["spe"] += avg_specificity_rbf
            result_dict[i + 1]["RBF"]["sen"] += avg_sensitivety_rbf



    # Hack central yo

    #calculate average
    for i in range(nr_of_folds):
        for classifier in result_dict[i]:
            for entry in ["ang", "margin", "acc", "spe", "sen"]:
                result_dict[i][classifier][entry] = result_dict[i][classifier][entry] / (K * (i + 1))
            for entry in ["TP", "TN", "FP", "FN"]:
                result_dict[i][classifier][entry] = sum(result_dict[i][classifier][entry]) / (K * (i + 1))

    result_dict[0]["SVM"] = result_dict[1]["SVM"]
    result_dict[0]["RBF"] = result_dict[0]["SVM"]
    result_dict[0]["HPF"] = result_dict[0]["SVM"]


    for measurement in measurements:
        result_file = open(measurement + ".csv", "w+")
        raw_file = open(measurement + "_raw" + ".data", "w+")
        write_data_to_file(result_dict, time_dict, result_file, [measurement], [])
        dump_raw_data(result_dict, raw_file)
        result_file.close()
        raw_file.close()

    for measurement in measurements:
        plot2d_from_columns(measurement + ".csv")

    plt.show();
