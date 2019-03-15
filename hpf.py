from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.datasets import make_blobs


import matplotlib.pyplot as plt
import numpy as np
import math
import json
from dimred import DR


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

    def ordering_support(self, vectors):
        """
        Returns the first possible primary support vector
        """
        w = self.clf.coef_[0]

        # As normal for the line is W = (b, -a)
        # direction is then given by as a = (-(-a), b))

        a = -w[1]
        b = w[0]

        cd = (vectors[0][0][:2] - vectors[1][0][:2]) / 2


        c = cd[0]
        d = cd[1]

        tk = []

        for key in vectors:
            for vector in vectors[key]:
                tk.append((((a * (vector[0] - c) + b * (vector[1] - d)) /
                                ( a * a + b * b)), vector, key))

        tk.sort(key=lambda x: x[0])

        return tk


    def left_or_right_of_plane(self, point, primary_support):
        w = self.clf.coef_[0]

        a = -w[0] / w[1]

        t = (point[0][0] - primary_support[0])/(-a)

        if (point[0][1]) <= (primary_support[1] + t):
            return 1
        else:
            return 0



    def split_data(self, primary_support):
        """
        returns a list  containing left and right split.
        """
        # Construct a new array, to remove reference
        right_set = [np.array(vector) for vector in zip(self.data_points, self.data_labels) if self.left_or_right_of_plane(vector, primary_support)]
        left_set = [np.array(vector) for vector in zip(self.data_points, self.data_labels) if not self.left_or_right_of_plane(vector, primary_support)]


        right_x = []
        right_y = []

        # Split data and labels into different lists
        for pair in right_set:
            right_x.append(pair[0])
            right_y.append(pair[1])


        left_x = []
        left_y = []

        for pair in left_set:
            left_x.append(pair[0])
            left_y.append(pair[1])


        return [[left_x, left_y], [right_x, right_y]]

    def get_splitting_point(self):
        """
        Finds and returns the primary support vector, splitting point
        """

        tk = self.ordering_support(self.support_vectors_dictionary)
        first_class = tk[0][2]

        primary_support_vector = None

        for vector in tk:
            if (vector[2] is not first_class):
                primary_support_vector = vector[1]

        return primary_support_vector

    def group_support_vectors(self):
        """
        groups the support vectors in the currently trained clf
        """
        self.support_vectors_dictionary = {}

        for sv in self.clf.support_vectors_:

            key = self.clf.predict([sv])[0]

            if key not in self.support_vectors_dictionary:
                self.support_vectors_dictionary[key] = [sv]
            else:
                self.support_vectors_dictionary[key].append(sv)



    def get_rotation(self, alpha):
        theta = alpha
        c, s = np.cos(theta), np.sin(theta)
        return np.array(((c,-s), (s, c)))

    def rotate_point_2D(self, point, angle, primary_support, intersection_point):
        """
        Returns the point, with it's xy-coordinates rotated accordingly to rubberband folding

        Does currently not apply rubberband folding, rotates points around intersection
        """
        rotation_matrix = self.get_rotation(angle)

        #point = np.matmul(point.T - intersection_point, rotation_matrix) + intersection_point

        #Slica föri helvete inte point i tilldelning här
        point = self.rot_func(point[:2], intersection_point, rotation_matrix)

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
            right_set[0] = [self.rotate_point_2D(point, angle, primary_support,
                            intersection_point)
                                for point in right_set[0]]
            left_or_right = 0

        elif (left_margin > right_margin):
            left_set[0] = [self.rotate_point_2D(point, -angle, primary_support,
                            intersection_point)
                                for point in left_set[0]]
            left_or_right = 1

        else:
            print("Cannot improve margin")


        X = left_set[0] + right_set[0]
        y = left_set[1] + right_set[1]


        #X = np.vstack(X)

        return (X, y, left_or_right, intersection_point)


    def fold(self):
        # folding sets
        right_clf = svm.SVC(kernel='linear', C=1000)
        left_clf = svm.SVC(kernel='linear', C=1000)


        # Splitting point
        self.primary_support_vector = self.get_splitting_point()

        # Used for plotting where the split occoured
        splitting_point = self.primary_support_vector[0]#x-axis-location of primary vec

        # Subsets of datasets, left and right of primary support vector
        left_set, right_set = self.split_data(self.primary_support_vector)

        # New SVM, right
        try:
            right_clf.fit(right_set[0], right_set[1])
            left_clf.fit(left_set[0], left_set[1])
        except ValueError:
            print("WARNING, ONLY ONE CLASS PRESENT IN A SET, ABORTING")
            return -1

        # Rotate and merge data sets back into one
        self.data_points, self.data_labels, left_or_right, intersection_point = self.rotate_set(left_clf, left_set, right_clf, right_set, self.primary_support_vector)

        self.rotation_data.append((intersection_point, self.primary_support_vector, left_or_right, (right_clf, left_clf)))

        # merge
        self.clf = right_clf if left_or_right else left_clf
        self.group_support_vectors()

        # Used for highlighting the sets
        right_set[0] = np.vstack(right_set[0])
        left_set[0] = np.vstack(left_set[0])
        return 0

    def classify(self, points, rotate=True):
        if not rotate:
            return self.old_clf.predict(points)

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

            rotated_set = [self.rotate_point_2D(point, angle, primary_support_vector, intersection_point) for point in rotated_set]

            points = np.asarray(rotated_set + none_rotated_set)


        return self.clf.predict(points)



        def get_distance_from_line_to_point(self, w, point, point_on_line):
            v = point - point_on_line
            proj = vector_projection(v, w)
            distance = np.linalg.norm(v - proj)

            return distance


    def clean_set(self):
        """
        Returns a cleaned dataset, turns soft into hard.

        Function does not calculate margin as get_margin does, something could be
        wrong with the assumption that len w is the margin. Instead it calculates
        the margin by calculating the distance from the decision_function to
        a support vector.
        """

        w = self.clf.coef_[0]
        w = np.array(w[1], w[0]) # place the vector in the direction of the line

        # Orginal support vectors
        point_on_line = (self.support_vectors_dictionary[0][0] + self.support_vectors_dictionary[1][0]) / 2

        margin = get_distance_from_line_to_point(w, self.support_vectors_dictionary[0][0], point_on_line)

        for point in self.data_points:
            distance = get_distance_from_line_to_point(w, point, point_on_line)

            if (distance < margin):
                index = np.where(self.data_points==point)

                self.data_points = np.delete(self.data_points, index, 0)

    def get_distance_from_line_to_point(self, w, point, point_on_line):
        v = point - point_on_line
        proj = vector_projection(v, w)
        distance = np.linalg.norm(v - proj)

        return distance


    def clean_set(self):
        """
        Returns a cleaned dataset, turns soft into hard.

        Function does not calculate margin as get_margin does, something could be
        wrong with the assumption that len w is the margin. Instead it calculates
        the margin by calculating the distance from the decision_function to
        a support vector.
        """

        w = self.clf.coef_[0]
        w = np.array(w[1], w[0]) # place the vector in the direction of the line

        # Orginal support vectors
        support_dict = group_support_vectors(clf.support_vectors_, clf)

        point_on_line = (support_dict[0][0] + support_dict[1][0]) / 2

        margin = get_distance_from_line_to_point(w, support_dict[0][0], point_on_line)

        for point in self.data_points:
            distance = get_distance_from_line_to_point(w, point, point_on_line)

            if (distance < margin):
                index = np.where(self.data_points==point)

                self.data_points = np.delete(self.data_points, index, 0)

                self.data_labels = np.delete(self.data_labels, index)




    def fit(self, data_points, data_labels):
        self.data_points = data_points
        self.data_labels = data_labels

        self.old_data = data_points

        self.clf.fit(data_points, data_labels)
        self.old_clf.fit(data_points, data_labels)
        self.old_margin = self.get_margin(self.old_clf)
        self.new_margin = -1
        #group into classes = create support_vectors_dictionary
        self.group_support_vectors()

        #project onto 2D
        #self.data_points, self.support_vectors_dictionary = self.dim_red.project_down(self.data_points, self.support_vectors_dictionary)


        #fold until just two support vectors exist or max_nr_of_folds is reached
        current_fold = 0
        val = 0
        while(len(self.clf.support_vectors_) > 2 and val is 0):
                val = self.fold()
                self.new_margin = self.get_margin(self.clf)
                current_fold += 1
                print(self.new_margin)


        if self.verbose:
            print("Number of folds: {}".format(current_fold))
            print("Margin change: {}".format(self.new_margin - self.old_margin))
            if current_fold > 0:
                for rotation in self.rotation_data:
                    intersection_point, primary_support_vector, left_or_right, clfs = rotation
                    print("intersection point: {}".format(intersection_point))
                    print("primary_support_vector {}".format(primary_support_vector))
                    print("Left or right: {}".format(left_or_right))

                    left_clf, right_clf = clfs

                    print("margin of left clf: {}".format(self.get_margin(left_clf)))

                    print("margin of right clf: {}".format(self.get_margin(right_clf)))
            else:
                print("Only two support vectors, no folds")


        #self.data_points = self.dim_red.project_up(self.data_points)

        stopper = 0


    def __init__(self,rot_func = lambda p, i, r : np.matmul(p.T - i, r) + i, max_nr_of_folds = 1, verbose = False):
        self.verbose = verbose
        self.max_nr_of_folds = max_nr_of_folds
        self.clf = svm.SVC(kernel='linear', C=1000)
        self.old_clf = svm.SVC(kernel='linear', C=1000)
        self.rotation_data = []
        self.rot_func = rot_func
        self.dim_red = DR()


    def __gr__(self, other):
        return self.new_margin - self.old_margin < other.new_margin - self.old_margin

    def json(self):
        dict = {}
        dict["margin"] = self.new_margin
        dict["old_margin"] = self.old_margin
        dict["increase_margin"] = self.new_margin - self.old_margin
        dict = json.dumps(dict)
        dict = json.loads(dict)
        return dict
