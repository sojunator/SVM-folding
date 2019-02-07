from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt
import numpy as np


def ordering_support(vectors, point, weights):
    """
    Returns the first possible primary support vector
    """

    max_key = min(vectors, key= lambda x: len(vectors[x]))
    return max_key

def get_splitting_point(support_dict, W):
    """
    Finds and returns the primary support vector, splitting point
    """
    return support_dict[ordering_support(support_dict, (0,0), [])][0]

def split_data(primary_support, X, Y):
    """
    returns a tuple  containing left and right split.
    """

    right_set = [vector for vector in zip(X,Y) if vector[0][0] >= primary_support[0]]
    left_set = [vector for vector in zip(X,Y) if vector[0][0] < primary_support[0]]

    right_x = []
    right_y = []

    for pair in right_set:
        right_x.append(pair[0])
        right_y.append(pair[1])


    left_x = []
    left_y = []

    for pair in left_set:
        left_x.append(pair[0])
        left_y.append(pair[1])


    return ((left_x, left_y), (right_x, right_y))

def group_support_vectors(support_vectors):
    """
    returns a dict containing lists of dicts, where key corresponds to class
    """
    # contains a dict of support vectors and class
    support_dict = {}

    for vector in support_vectors:
        key = clf.predict([vector])[0]

        if key not in support_dict:
            support_dict[key] = [vector]
        else:
            support_dict[key].append(vector)

    return support_dict

if __name__ == "__main__":
    # Create classifier

    X, y = make_blobs(n_samples=40, centers=2, random_state=6)

    # fit the model, don't regularize for illustration purposes
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X, y)


    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    # plot the decision function




    support_dict = group_support_vectors(clf.support_vectors_)

    primary_support = get_splitting_point(support_dict, [])
    left_set, right_set = split_data(primary_support, X, y)

    clf.fit(right_set[0], right_set[1])

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    plt.axvline(x=primary_support[0])
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')


    plt.show()
