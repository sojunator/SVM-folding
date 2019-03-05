from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt
import numpy as np
import math
import json


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

        #point = np.matmul(point.T - intersection_point, rotation_matrix) + intersection_point
        point = self.rot_func(point, intersection_point, rotation_matrix)
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

    def grahm_schmidt_orthonorm(self, linearly_independent_support_vectors):

        orthonormated_vectors = []#stores the new basis

        vec = linearly_independent_support_vectors[0]
        vec = vec / np.linalg.norm(vec)#first entry is just the itself normalized
        orthonormated_vectors.append(vec)

        i = 0
        for v in linearly_independent_support_vectors[1:]:

            vec = 0
            for u in orthonormated_vectors:
                projection = np.dot(v,u) * u
                projection[np.abs(projection) < 0.000001] = 0
                vec -= projection

            vec = v + vec
            vec[np.abs(vec) < 0.000001] = 0

            if all(v == 0 for v in vec):#if true then this vector is dependent on it's predicessors
                print(i)

            i += 1

            vec = vec / np.linalg.norm(vec)
            orthonormated_vectors.append(vec)

        return orthonormated_vectors

    def linind(self, support_vectors):#asdfasdfc

        linearly_independent_support_vectors = support_vectors

        progresser = 0
        i = 0
        while i < len(linearly_independent_support_vectors) - 1:

            li_vec1 = linearly_independent_support_vectors[i]

            linearlyDependent = False
            dependentIndexes = np.ones(len(linearly_independent_support_vectors), dtype=bool)



            for k in range(progresser + 1, len(linearly_independent_support_vectors)):

                li_vec2 = linearly_independent_support_vectors[k]

                if cauchy_schwarz_equal(li_vec1, li_vec2):#if true, li_vec1 and li_vec2 are dependent according to cauchy-schwarz-inequality
                    dependentIndexes[k] = False
                    linearlyDependent = True


            if linearlyDependent:
                linearly_independent_support_vectors = linearly_independent_support_vectors[dependentIndexes]# remove vectors that were dependent with li_vec1
            else:
                progresser += 1 # increment 'progresser', since vec[i] is independent

            i = progresser # reset iteration

        return linearly_independent_support_vectors

    def cauchy_schwarz_equal(self, v1, v2):
        """
        returns true if the cauchy-schwarz inequality is equal, meaning that they are linearly dependent
        if the function returns true, the vectors are linearly dependent, and one of the vectors can be removed
        """

        #inner product
        ipLeft = np.dot(v1, v2)

        return ipLeft * ipLeft == np.dot(v1,v1) * np.dot(v2,v2)

    def get_linearly_independent_support_vectors(self, support_vectors):

        linearly_independent_support_vectors = support_vectors

        progresser = 0
        i = 0
        while i < len(linearly_independent_support_vectors) - 1:

            li_vec1 = linearly_independent_support_vectors[i]

            linearlyDependent = False
            dependentIndexes = np.ones(len(linearly_independent_support_vectors), dtype=bool)


            for k in range(progresser + 1, len(linearly_independent_support_vectors)):

                li_vec2 = linearly_independent_support_vectors[k]

                if cauchy_schwarz_equal(li_vec1, li_vec2):#if true, li_vec1 and li_vec2 are dependent according to cauchy-schwarz-inequality
                    dependentIndexes[k] = False
                    linearlyDependent = True


            if linearlyDependent:
                linearly_independent_support_vectors = linearly_independent_support_vectors[dependentIndexes]# remove vectors that were dependent with li_vec1
            else:
                progresser += 1 # increment 'progresser', since vec[i] is independent

            i = progresser # reset iteration


        return linearly_independent_support_vectors

    def align_axis(self, support_vectors):

        linearly_independent_support_vectors = 0

        orthonormated_basis = grahm_schmidt_orthonorm(linearly_independent_support_vectors)

        return

    def get_direction_between_two_vectors_in_set_with_smallest_distance(self, set, dim):
        """
        Finds the shortest distance between two vectors within the given set.
        """
        if (len(set) <2):
            print("Error, less than two support vectors in set")
            return

        bestDir = set[0] - set[1]
        bestDist = np.linalg.norm(bestDir)

        for index_v1 in range(0, len(set)):
            vec1 = set[index_v1]
            for vec2 in set[index_v1 + 1:]:

                dir = vec1 - vec2
                dist = np.linalg.norm(dir)
                if dist < bestDist:#found two vecs with shorter distance inbetween
                    bestDist = dist
                    bestDir = dir

        return bestDir[:dim]

    def get_rotation_matrix_onto_lower_dimension(self, support_vectors_from_one_class, dim):
        """
        Forms a lower triangular rotation matrix
        In the function, 'diagonal' is NOT denoted as the 'center'-diagonal. It is selected as: matrix[row][row+1] for a row-major matrix
        """

        rotation_matrix = np.zeros((dim,dim))

        #d is the shortest direction between two support vectors in one of the classes
        dir = get_direction_between_two_vectors_in_set_with_smallest_distance(support_vectors_from_one_class, dim)

        #Wk = sqrt(v1^2 + v2^2 ... + vk^2)
        squaredElementsAccumulator = dir[0] * dir[0] + dir[1] * dir[1]

        Wk = dir[0]#for k = 1
        Wkp1 = np.sqrt(squaredElementsAccumulator)

        #first row
        if Wkp1 != 0:
            rotation_matrix[0][0] = dir[1] / Wkp1#first element
            rotation_matrix[0][1] = -Wk / Wkp1#first diagonal element
        else:
            rotation_matrix[0][0] = 1#first element
            rotation_matrix[0][1] = 0#first diagonal element


        #middle rows
        for row in range(1, dim - 1):

            diagonalElement = dir[row + 1]#row + 1 is the k'th element in the vector
            squaredElementsAccumulator += diagonalElement * diagonalElement#accumulate next step, square next element

            Wk = Wkp1
            Wkp1 = np.sqrt(squaredElementsAccumulator)

            #diagonal entry in matrix
            U = 0
            if Wkp1 != 0:
                U = Wk / Wkp1

            rotation_matrix[row][row + 1] = -U

            #denominator per row
            denominator = Wk * Wkp1

            if denominator == 0:
                rotation_matrix[row][row] = 1

            else:
                i = 0
                for element in dir[0:row+1]:
                    rotation_matrix[row][i] = element*diagonalElement / denominator
                    i+=1

        #last row in matrix
        if Wkp1 != 0:
            rotation_matrix[dim-1] = [element / Wkp1 for element in dir]
        else:
            rotation_matrix[dim-1][dim-1] = 1


        return rotation_matrix

    def dimension_projection(self, support_dict):
        #Input: full dataset for a clf, and support vectors separated into classes in a dictionary
        #if supportvectors  -> k < n + 1 run align axis aka if(n <= currentDim) then -> align_axis
        #align_axis

        rotDim = dim = len(support_dict[0][0])

        while rotDim > 2:
            support_vectors_from_one_class = 0
            if (len(support_dict[0]) > len(support_dict[1])):#pick class with most vectors in
                support_vectors_from_one_class = support_dict[0]
            else:
                support_vectors_from_one_class = support_dict[1]

            rotation_matrix = get_rotation_matrix_onto_lower_dimension(support_vectors_from_one_class, rotDim)

            #rotate all datapoint
            rotDim -= 1
            self.data_points = [np.matmul(rotation_matrix, point)[:rotDim] for point in self.data_points]
            support_dict[0] = [np.matmul(rotation_matrix, point)[:rotDim] for point in support_dict[0]]
            support_dict[1] = [np.matmul(rotation_matrix, point)[:rotDim] for point in support_dict[1]]


        return dataset, support_dict

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

    def create_classifier(self):
        while(len(self.clf.support_vectors_) > 2):
            self.fold()
            self.new_margin = self.get_margin(self.clf)


    def __init__(self, data_points, data_labels, rot_func = lambda p, i, r : np.matmul(p.T - i, r) + i):
        self.data_points = data_points
        self.data_labels = data_labels
        self.clf = svm.SVC(kernel='linear', C=1000)
        self.old_clf = svm.SVC(kernel='linear', C=1000)
        self.rotation_data = []
        self.rot_func = rot_func


        self.clf.fit(data_points, data_labels)
        self.old_clf.fit(data_points, data_labels)
        self.old_margin = self.get_margin(self.old_clf)
        self.support_vectors = self.clf.support_vectors_


        self.create_classifier()

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
