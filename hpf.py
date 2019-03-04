from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt
import numpy as np
import math


class HPF:
    def get_rotation(self, alpha):
        theta = alpha
        c, s = np.cos(theta), np.sin(theta)
        return np.array(((c,-s), (s, c)))

    def get_hyperplane(self, clf):
        """
        Returns hyperplane for classifer
        """

        w = clf.coef_[0]
        a = -w[0] / w[1]

        return (a, (-clf.intercept_[0]) / w[1])

    def get_intersection_point(self, left, right):
        """
        Takes two sklearn svc classifiers that are trained on subsets of the same
        dataset

        Returns touple of intersection point and intersection angle alpha
        ((x,y), alpha)
        """

        # get hyperplanes
        left_hyperplane, right_hyperplane = self.get_hyperplane(left), self.get_hyperplane(right)
        x = (left_hyperplane[1] - right_hyperplane[1]) / (right_hyperplane[0] - left_hyperplane[0])

        y = right_hyperplane[0] * x + right_hyperplane[1]

        angle = np.arctan(right_hyperplane[0]) - np.arctan(left_hyperplane[0])
        return ((x, y), angle)


    def get_margin(self, clf):
        """
        https://scikit-learn.org/stable/auto_examples/svm/plot_svm_margin.html
        returns the margin of given clf
        """

        return 1 / np.sqrt(np.sum(clf.coef_ ** 2))

    def split_data(self, primary_support):
        """
        returns a list  containing left and right split.
        """

        right_set = [vector for vector in zip(self.data_points, self.data_labels) if vector[0][0] >= primary_support[0]]
        left_set = [vector for vector in zip(self.data_points, self.data_labels) if vector[0][0] <= primary_support[0]]

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

    def ordering_support(self, vectors):
        """
        Returns the first possible primary support vector
        """
        w = self.clf.coef_[0]

        # As normal for the line is W = (b, -a)
        # direction is then given by as a = (-(-a), b))

        a = -w[1]
        b = w[0]

        cd = (vectors[0][0] - vectors[1][0]) / 2

        c = cd[0]
        d = cd[1]

        tk = []

        for key in vectors:
            for vector in vectors[key]:
                tk.append((((a * (vector[0] - c) + b * (vector[1] - d)) /
                                ( a * a + b * b)), vector, key))

        tk.sort(key=lambda x: x[0])

        return tk

    def get_splitting_point(self, support_dict):
        """
        Finds and returns the primary support vector, splitting point
        """

        tk = self.ordering_support(support_dict)

        first_class = tk[0][2]

        primary_support_vector = None

        for vector in tk:
            if (vector[2] is not first_class):
                primary_support_vector = vector[1]

        return primary_support_vector

    def group_support_vectors(self):
        """
        returns a dict containing lists of dicts, where key corresponds to class
        """
        # contains a dict of support vectors and class
        support_dict = {}

        for vector in self.support_vectors:
            key = self.clf.predict([vector])[0]

            if key not in support_dict:
                support_dict[key] = [vector]
            else:
                support_dict[key].append(vector)

        return support_dict


    def get_rotation(self, alpha):
        theta = alpha
        c, s = np.cos(theta), np.sin(theta)
        return np.array(((c,-s), (s, c)))

    def rotate_point(self, point, angle, primary_support, intersection_point):
        """
        Returns the point rotated accordingly to rubberband folding

        Does currently not apply rubberband folding, rotates points around intersection
        """
        rotation_matrix = self.get_rotation(angle)

        point = np.matmul(point.T - intersection_point, rotation_matrix) + intersection_point

        return point


    def rotate_set(self, left_clf, left_set, right_clf, right_set, primary_support):
        """
        Performs rotation on the set with biggest margin
        Currently rotates around the intersection point

        Does not contain datapoints to set

        returns a merged and rotated set, touple (X, y)
        """

        # Get margins
        right_margin = self.get_margin(right_clf)
        left_margin = self.get_margin(left_clf)


        # intersection data
        intersection_point, angle = self.get_intersection_point(left_clf, right_clf)

        # if 1, left was rotated, 0 is right set.
        left_or_right = -1

        if (right_margin > left_margin):
            right_set[0] = [self.rotate_point(point, angle, primary_support, intersection_point)
                                for point in right_set[0]]
            left_or_right = 0

        elif (left_margin > right_margin):
            left_set[0] = [self.rotate_point(point, -angle, primary_support, intersection_point)
                                for point in left_set[0]]
            left_or_right = 1

        else:
            print("Cannot improve margin")


        X = left_set[0] + right_set[0]

        y = left_set[1] + right_set[1]


        X = np.vstack(X)

        return (X, y, left_or_right, intersection_point)


    def fold(self):
        # folding sets
        right_clf = svm.SVC(kernel='linear', C=1000)
        left_clf = svm.SVC(kernel='linear', C=1000)

        # Orginal support vectors
        support_dict = self.group_support_vectors()

        # Splitting point
        primary_support_vector = self.get_splitting_point(support_dict)

        # Used for plotting where the split occoured
        splitting_point = primary_support_vector[0]#x-axis-location of primary vec

        # Subsets of datasets, left and right of primary support vector
        left_set, right_set = self.split_data(primary_support_vector)

        # New SVM, right
        right_clf.fit(right_set[0], right_set[1])
        left_clf.fit(left_set[0], left_set[1])

        # Rotate and merge data sets back into one
        self.data_points, self.data_labels, left_or_right, intersection_point = self.rotate_set(left_clf,
                                                                                left_set,
                                                                                right_clf,
                                                                                right_set,
                                                                                primary_support_vector)

        self.rotation_data.append((intersection_point, primary_support_vector, left_or_right, (right_clf, left_clf)))

        # merge
        self.clf = right_clf if left_or_right else left_clf
        self.support_vectors = self.clf.support_vectors_

        # Used for highlighting the sets
        right_set[0] = np.vstack(right_set[0])
        left_set[0] = np.vstack(left_set[0])


    def classify(self, points):
        for rotation in self.rotation_data:
            #unpackage the mess
            intersection_point = rotation[0]
            primary_support_vector = rotation[1]
            left_or_right = rotation[2]
            right_clf = rotation[3][0]
            left_clf = rotation[3][1]

            _, angle = self.get_intersection_point(left_clf, right_clf)
            rotation_matrix = self.get_rotation(angle)

            rotated_set = []
            none_rotated_set = []


            for point in points:
                if left_or_right:
                    if primary_support_vector[0] >= point[0]:
                        rotated_set.append(point)
                    else:
                        none_rotated_set.append(point)
                else:
                    if primary_support_vector[0] <= point[0]:
                        rotated_set.append(point)
                    else:
                        none_rotated_set.append(point)

            rotated_set = [self.rotate_point(point, angle, primary_support_vector, intersection_point) for point in rotated_set]

            points = np.asarray(rotated_set + none_rotated_set)


        return self.clf.predict(points)

    def create_classifier(self):
        while(len(self.clf.support_vectors_) > 2):
            self.fold()
            print(self.get_margin(self.clf))


    def __init__(self, data_points, data_labels):
        self.data_points = data_points
        self.data_labels = data_labels
        self.clf = svm.SVC(kernel='linear', C=1000)
        self.rotation_data = []

        self.clf.fit(data_points, data_labels)

        self.support_vectors = self.clf.support_vectors_


        self.create_classifier()
