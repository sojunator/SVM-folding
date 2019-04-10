from hpf import HPF
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import pandas as pd



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
