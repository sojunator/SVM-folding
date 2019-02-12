from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt
import numpy as np
import math

def rotate_set(left_clf, left_set, right_clf, right_set):
    """
    Performs rotation on the set with biggest margin
    Currently rotates around the intersection point

    Does not contain datapoints to set

    returns a merged and rotated set
    """
    # Get margins
    right_margin = get_margin(right_clf)
    left_margin = get_margin(left_clf)

    # intersection data
    intersection_point, angle = get_intersection_point(left_clf, right_clf)
    rotation_matrix = get_rotation(angle)

    if (right_margin > left_margin):
        right_set[0] = [np.matmul((point.T - intersection_point), rotation_matrix) +  intersection_point for point in right_set[0]]

    elif (left_margin > right_margin):
        left_set[0] = [np.matmul(point.T - intersection_point, rotation_matrix) + - intersection_point for point in left_set[0]]
    else:
        print("Cannot improve margin")

    X = left_set[0] + right_set[0]
    y = left_set[1] + right_set[1]

    X = np.vstack(X)

    return (X, y)

def get_rotation(alpha):
    theta = alpha
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c,-s), (s, c)))

def get_margin(clf):
    """
    https://scikit-learn.org/stable/auto_examples/svm/plot_svm_margin.html
    returns the margin of given clf
    """

    return 1 / np.sqrt(np.sum(clf.coef_ ** 2))

def get_hyperplane(clf):
    """
    Returns hyperplane for classifer
    """

    w = clf.coef_[0]
    a = -w[0] / w[1]

    return (a, (-clf.intercept_[0]) / w[1])

def get_intersection_point(left, right):
    """
    Takes two sklearn svc classifiers that are trained on subsets of the same
    dataset

    Returns touple of intersection point and intersection angle alpha
    ((x,y), alpha)
    """

    # get hyperplanes
    left_hyperplane, right_hyperplane = get_hyperplane(left), get_hyperplane(right)
    x = (left_hyperplane[1] - right_hyperplane[1]) / (right_hyperplane[0] - left_hyperplane[0])

    y = right_hyperplane[0] * x + right_hyperplane[1]

    angle = np.arctan(right_hyperplane[0]) - np.arctan(left_hyperplane[0])
    return ((x, y), angle)

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
    left_set = [vector for vector in zip(X,Y) if vector[0][0] <= primary_support[0]]

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


    return [[left_x, left_y], [right_x, right_y]]

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


def plot(clf, left_clf, right_clf):
    """
    God function that removes all the jitter from main
    """

    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])


    ax.scatter(right_set[0][:, 0], right_set[0][:, 1], s=100,
       linewidth=1, facecolors='none', edgecolors='b')


    ax.scatter(left_set[0][:, 0], left_set[0][:, 1], s=100,
       linewidth=1, facecolors='none', edgecolors='r')




    # right support vectors
    ax.scatter(right_clf.support_vectors_[:, 0], right_clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='b')



    Z = right_clf.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='b', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # left support vectors
    ax.scatter(left_clf.support_vectors_[:, 0], left_clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='r')

    Z = left_clf.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='r', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    plt.show()

if __name__ == "__main__":
    # Dataset
    X, y = make_blobs(n_samples=40, centers=2, random_state=6)

    # Original SVM
    clf = svm.SVC(kernel='linear', C=1000)
    # folding sets
    right_clf = svm.SVC(kernel='linear', C=1000)
    left_clf = svm.SVC(kernel='linear', C=1000)

    # Train on inital data
    clf.fit(X, y)

    print("Old margin {}".format(get_margin(clf)))



    # Orginal support vectors
    support_dict = group_support_vectors(clf.support_vectors_)

    # Splitting point
    primary_support = get_splitting_point(support_dict, [])

    # Subsets of datasets, left and right of primary support vector
    left_set, right_set = split_data(primary_support, X, y)

    # New SVM, right
    right_clf.fit(right_set[0], right_set[1])
    left_clf.fit(left_set[0], left_set[1])


    # Rotate and merge data sets back into one
    X, y = rotate_set(left_clf, left_set, right_clf, right_set)

    # merge
    clf.fit(X,y)

    print("New margin {}".format(get_margin(clf)))

    # Used for highlighting the sets
    right_set[0] = np.vstack(right_set[0])
    left_set[0] = np.vstack(left_set[0])

    plot(clf, left_clf, right_clf)
