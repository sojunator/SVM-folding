from hpf import HPF
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import pandas as pd

def plot_clf(clf, ax, XX, YY, colour='k'):
    """
    Plots a clf, with margins, colour will be black
    """

    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors=colour)

    ax.contour(XX, YY, Z, colors=colour, levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])


def plot(hpf):
    """
    God function that removes all the jitter from main
    """
    X = hpf.data_points
   
    X = [x[:2] for x in X]

    

    plt.scatter(X[0][:], X[1][:], c=[0,1], s=30, cmap=plt.cm.Paired)
    ax = plt.gca()
    xlim = (-2, 50)
    ylim = (-2, -51)

    #plt.axvline(x=splitting_point, color='k')

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)

    YY, XX = np.meshgrid(yy, xx)


    plot_clf(hpf.clf, ax, XX, YY, 'g')
    plot_clf(hpf.old_clf, ax, XX, YY, 'k')


    plt.show()

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
            data_labels = np.array([row[-1] for row in temp])
            datasets[file] = (data_points, data_labels)

    return datasets
