from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Create classifier
    fignum = 1
    clf = svm.SVC(kernel="linear")

    data, target = datasets.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)


    X_train, X_test, y_train, y_test = train_test_split(data,
                                                        target,
                                                        test_size=0.0,# 70% training and 30% test
                                                        random_state=109)
    clf.fit(X_train, y_train)
    support_vectors = clf.support_vectors_
    print(len(support_vectors))

    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)

    yy = a * xx - (clf.intercept_[0]) / w[1]

    print(" w {}     a {} ".format(w, a))

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors (margin away from hyperplane in direction
    # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
    # 2-d.
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')


    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')

    plt.scatter(data[:, 0], data[:, 1], marker='o', c=target,
                s=25, edgecolor='k')
    plt.show()
