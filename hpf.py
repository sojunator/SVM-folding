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



def vec_equal(vec1, vec2):

    return np.allclose(vec1, vec2)

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

class HPF:
    def plot_self(self, new_figure = False):
        if new_figure:
            plt.figure()
        p = (self.support_vectors_dictionary[0][0][:2] + self.support_vectors_dictionary[1][0][:2]) / 2
        n = self.hyperplane_normal
        h = [0,0]
        h[1] = n[0]
        h[0] = -n[1]

        hx1 = p[0] + h[0] * 100
        hy1 = p[1] + h[1] * 100

        hx2 = p[0] + h[0] * -100
        hy2 = p[1] + h[1] * -100

        plt.plot([hx1,hx2],[hy1, hy2], 'b')

        x1 = []
        y1 = []
        x2 = []
        y2 = []

        for index, label in enumerate(self.data[1]):
            #print(index)
            if label == 0:

                x1.append(self.data[0][index][0])
                y1.append(self.data[0][index][1])

            elif label == 1:
                x2.append(self.data[0][index][0])
                y2.append(self.data[0][index][1])


        plt.plot(x1, y1, 'ro')
        plt.plot(x2, y2, 'go')

        #plt.axis([-10, 10, -10, 10])




    def vector_projection(self, v1, v2):
        """
        v1 projected on v2
        """
        return np.dot(v1, v2) / np.dot(v2,v2) * v2

    def get_hyperplane_normal(self):
        """
        Returns the normalized normal for the current clf
        """
        w = self.clf.coef_[0]

        n = w / np.linalg.norm(w)

        return n



    def get_intersection_between_SVMs(self, left, right):
        """
        http://www.ambrsoft.com/MathCalc/Line/TwoLinesIntersection/TwoLinesIntersection.htm

        Takes two sklearn svc classifiers that are trained on subsets of the same
        dataset

        Returns touple of intersection point and intersection angle alpha
        ((x,y), alpha)
        """

        # get hyperplanes
        w_1 = left.coef_[0]
        w_2 = right.coef_[0]

        c1 = left.intercept_[0]
        c2 = right.intercept_[0]

        x = (w_1[1] * c2 - w_2[1] * c1) / (w_1[0] * w_2[1] - w_2[0] * w_1[1])
        y = (w_2[0] * c1 - w_1[0] * c2) / (w_1[0] * w_2[1] - w_2[0] * w_1[1])

        angle =  w_1[0] * w_2[0] + w_1[1] * w_2[1]
        angle = angle / np.sqrt((w_1[0] * w_1[0] + w_1[1] * w_1[1]) * (w_2[0] * w_2[0] + w_2[1] * w_2[1]))
        #angle = np.arccos(angle)


        return ((x, y), angle)




    def get_margin(self, clf):
        """
        https://scikit-learn.org/stable/auto_examples/svm/plot_svm_margin.html
        returns the margin of given clf
        """

        return 1 / np.sqrt(np.sum(clf.coef_ ** 2))



    def ordering_support(self):
        """
        Returns the first possible primary support vector
        """
        n = self.hyperplane_normal

        # As normal for the line is W = (b, -a)
        # direction is then given by as a = (-(-a), b))

        a = n[1]
        b = -n[0]

        #point on line
        cd = (self.support_vectors_dictionary[0][0][:2] + self.support_vectors_dictionary[1][0][:2]) / 2


        c = cd[0]
        d = cd[1]

        tk = []

        for key in self.support_vectors_dictionary:
            for vector in self.support_vectors_dictionary[key]:
                tk.append((((a * (vector[0] - c) + b * (vector[1] - d)) /
                                ( a * a + b * b)), vector, key))

        tk.sort(key=lambda x: x[0])

        return tk


    def left_or_right_of_plane(self, point, primary_support_vector = None):
        """
        Splits the data from the primary point in the direction of the normal
        """

        # Classify needs to overwrite sv
        if primary_support_vector is None:
            primary_support_vector = self.primary_support_vector[:2]

        n = self.hyperplane_normal

        # planes direction
        h = [0,0]
        h[0] = n[1]
        h[1] = -n[0]


        # vector between point and pv
        ppv = point[:2] - primary_support_vector

        # normalize
        h = h / np.linalg.norm(h)
        norm_ppv = np.linalg.norm(ppv)

        if norm_ppv < 0.00001: #is overlapping the primary support vector
            return True

        ppv = ppv / norm_ppv

        # angle between ppv and normal.
        cosang = np.dot(ppv, h)

        return cosang > 0.0

    def split_data(self, data = None, labels = None, primary_support_vector = None):
        """
        returns a list  containing left and right split.
        """
        # Construct a new array, to remove reference
        if (data is None and primary_support_vector is None):
            data = np.array(self.data[0])
            primary_support_vector = self.primary_support_vector
            labels = self.data[1]


        right_set = [[],[]]
        left_set = [[],[]]
        not_added = True

        for point, label in zip(data, labels):
            # Add primary support vector to both sets
            if np.allclose(point, primary_support_vector) and not_added:
                right_set[0].append(point)
                right_set[1].append(label)
                left_set[0].append(point)
                left_set[1].append(label)
                not_added = False

            # Vector is not primary, it should reside in one of the sets
            else:
                # Get which side the point resides on
                if self.left_or_right_of_plane(point):
                    left_set[0].append(point)
                    left_set[1].append(label)

                else:
                    right_set[0].append(point)
                    right_set[1].append(label)

        return left_set, right_set

    def get_splitting_point(self):
        """
        Finds and returns the primary support vector, splitting point
        """

        tk = self.ordering_support()
        first_class = tk[0][2]


        for vector in tk:
            if (vector[2] is not first_class):
                return vector[1]

        print("Error in spliting point: No primary vector found")

    def group_support_vectors(self, clf):
        """
        Input: a trained SVS
        Output: support vectors grouped into classes in a dictionary
        """
        grouped_support_vectors = {}

        for sv in clf.support_vectors_:

            key = clf.predict([sv])[0]

            if key not in grouped_support_vectors:
                grouped_support_vectors[key] = [sv]
            else:
                checks = [np.allclose(sv, point ) for point in grouped_support_vectors[key]]

                if True not in checks:
                    grouped_support_vectors[key].append(sv)


        return grouped_support_vectors

    def get_rotation(self, alpha):
        """
        Forms the rotation matrix:
        (cos, -sin)
        (sin, cos)
        """
        c = alpha
        s = np.sqrt(1 - alpha * alpha)
        #theta = alpha
        #c, s = np.cos(theta), np.sin(theta)
        return np.array(((c,-s), (s, c)))

    def get_counter_rotation(self, alpha):
        """
        Forms the counter clockwise rotation matrix:
        (cos, sin)
        (-sin, cos)
        """
        c = alpha
        s = np.sqrt(1 - alpha * alpha)
        #theta = alpha
        #c, s = np.cos(theta), np.sin(theta)
        return np.array(((c,s), (-s, c)))


    def get_rubber_band_angle(self, angle):

        return None

    def rotate_left(self, point, angle, intersection_point):


        rotation_matrix = self.get_counter_rotation(angle)

        x,y = self.rot_func(point[:2], intersection_point, rotation_matrix)

        point[0] = x
        point[1] = y

        return point

    def rotate_right(self, point, angle, intersection_point):


        rotation_matrix = self.get_rotation(angle)

        x,y = self.rot_func(point[:2], intersection_point, rotation_matrix)

        point[0] = x
        point[1] = y

        return point


    def rotate_point_2D(self, point, angle, primary_support, intersection_point, clf):
        """
        Returns the point, with it's xy-coordinates rotated accordingly to rubberband folding

        Does currently not apply rubberband folding, rotates points around intersection
        """
        norm = np.linalg.norm(clf.coef_[0])
        v = intersection_point + norm

        cosang = np.dot(v, intersection_point - point[:2])
        sinang = np.linalg.norm(np.cross(v, intersection_point - point[:2]))
        angle2 = np.arctan2(sinang, cosang)


        rotation_matrix = self.get_rotation(min(angle, angle))

        #point = np.matmul(point.T - intersection_point, rotation_matrix) + intersection_point

        x,y = self.rot_func(point[:2], intersection_point, rotation_matrix)
        point[0] = x
        point[1] = y

        return point


    def rotate_set(self, left_clf, left_set, right_clf, right_set, primary_support_vector):
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
        intersection_point, angle = self.get_intersection_between_SVMs(left_clf, right_clf)#self.get_intersection_point(left_clf, right_clf)
        # if 1, left was rotated, 0 is right set.

        #rotate left or right
        d = intersection_point - self.primary_support_vector[:2]
        d = d / np.linalg.norm(d)

        n = self.hyperplane_normal[:2]
        n = n / np.linalg.norm(n)

        c = np.dot(d, n)

        left_or_right = -1

        if c > 0.0:
            left_set[0] = [self.rotate_left(point, angle, intersection_point) for point in left_set[0]]
            left_or_right = 1

        else:
            left_set[0] = [self.rotate_right(point, angle, intersection_point) for point in left_set[0]]
            left_or_right = 0



        X = left_set[0] + right_set[0]
        y = left_set[1] + right_set[1]


        return (X, y, left_or_right, intersection_point)


    def fold(self):
        # folding sets
        right_clf = svm.SVC(kernel='linear', C=1e10)
        left_clf = svm.SVC(kernel='linear', C=1e10)


        # Splitting point
        self.primary_support_vector = self.get_splitting_point()

        # Subsets of datasets, left and right of primary support vector
        left_set, right_set = self.split_data()


        # New SVM, right
        try:
            #Reduce problem down to 2d
            right_2d = [np.array(point[:2]) for point in right_set[0]]
            left_2d = [np.array(point[:2]) for point in left_set[0]]


            right_clf.fit(right_2d, right_set[1])
            left_clf.fit(left_2d, left_set[1])

        except ValueError:

            print("WARNING, ONLY ONE CLASS PRESENT IN A SET, ABORTING")
            print(self.support_vectors_dictionary)
            print(self.primary_support_vector)
            self.plot_self()
            plt.show()
            return -1



        # Rotate and merge data sets back into one
        self.data[0], self.data[1], left_or_right, intersection_point = self.rotate_set(left_clf, left_set, right_clf, right_set, self.primary_support_vector)

        self.rotation_data.append((intersection_point, self.primary_support_vector, left_or_right, (right_clf, left_clf), self.support_vectors_dictionary))

        return 0

    def classify(self, points, rotate=True):
        if not rotate:
            return self.old_clf.predict(points)

        # Correct dim is used to save last down projection
        # As we don't want if-statement checking for last iteration
        correct_dim = []
        for idx, rotation in enumerate(self.rotation_data):
            points = self.dim_red.classify_project_down(points, idx)

            #unpackage the mess
            primary_support_vector = rotation[1]
            right_clf = rotation[3][0]
            left_clf = rotation[3][1]


            left_set, right_set = self.split_data(points, [None] * len(points), primary_support_vector)
            print("left set len: {}".format(len(left_set[0])))
            print("right set len: {}".format(len(right_set[0])))
            points, y, __, ___ = self.rotate_set(left_clf, left_set, right_clf, right_set, primary_support_vector)

            # On return will contain last down projection
            correct_dim = np.array(points)

            points = self.dim_red.classify_project_up(points, idx)


        return self.clf.predict(correct_dim)

    def plot_dir(self, dir, point, new_figure_ = False):
        if new_figure_:
            plt.figure()

        h = dir

        w = [0,0]
        w[0] = -h[1]
        w[1] = h[0]

        p = point

        plot_size = 10

        hx1 = p[0] + h[0] * plot_size
        hy1 = p[1] + h[1] * plot_size

        hx2 = p[0] + h[0] * -plot_size
        hy2 = p[1] + h[1] * -plot_size

        plt.plot([hx1,hx2],[hy1, hy2], 'g')

        hx1 = p[0] + w[0] * plot_size
        hy1 = p[1] + w[1] * plot_size

        hx2 = p[0] + w[0] * -plot_size
        hy2 = p[1] + w[1] * -plot_size

        plt.plot([hx1,hx2],[hy1, hy2], 'b')

    def plot_data(self, data, new_figure_ = False):

        if new_figure_:
            plt.figure()

        x1 = []
        y1 = []
        x2 = []
        y2 = []

        for index, label in enumerate(data[1]):

            if label == 0:

                x1.append(data[0][index][0])
                y1.append(data[0][index][1])

            elif label == 1:
                x2.append(data[0][index][0])
                y2.append(data[0][index][1])

        plt.plot(x1, y1, 'ro')
        plt.plot(x2, y2, 'go')

    def fit(self, data_points, data_labels):
        self.data = [data_points, data_labels]

        self.old_data = data_points

        # Builds clfs
        self.old_clf.fit(data_points, data_labels)
        self.clf.fit(data_points, data_labels)

        self.old_margin = self.get_margin(self.old_clf)
        self.new_margin = -1


        #project onto 2D
        self.current_fold = 0
        val = 0
        #group into classes = create support_vectors_dictionary
        self.support_vectors_dictionary = self.group_support_vectors(self.clf)
        self.hyperplane_normal = self.get_hyperplane_normal()

        # floating point termination
        previous_margin = self.old_margin
        margins = []

        while(len(self.clf.support_vectors_) > 2 and val is 0):
            self.data[0], self.support_vectors_dictionary, self.hyperplane_normal = self.dim_red.project_down(self.data[0], self.support_vectors_dictionary, self.hyperplane_normal)

            #self.plot_self(True)

            val = self.fold()

            #self.plot_self(True)

            #plt.show()
            if self.current_fold > 150:
                self.plot_self()
                plt.show()

            self.current_fold += 1

            self.data[0] = self.dim_red.project_up(self.data[0])



            #fit for next iteration or exit contidion of just two support vectors
            self.clf.fit(self.data[0], self.data[1])
            self.support_vectors_dictionary = self.group_support_vectors(self.clf) #regroup
            self.hyperplane_normal = self.get_hyperplane_normal()


            self.new_margin = self.get_margin(self.clf)
            margins.append(math.fabs(self.new_margin - previous_margin))
            previous_margin = self.new_margin

            if len(margins) == 3:
                avg = sum(margins) / len(margins)
                if avg < 0.001:
                    val = -1
                    print("termination due to floating point")
                margins.clear()


        print("nr of support {}".format(len(self.clf.support_vectors_)))


        #self.clf.fit(self.data_points, self.data_labels)
        #self.new_margin = self.get_margin(self.clf)
        self.new_margin = self.get_margin(self.clf)
        print(self.new_margin)
        print(self.old_margin)

        if self.verbose:
            print("Number of folds: {}".format(self.current_fold))
            print("Margin change: {}".format(self.new_margin - self.old_margin))
            if len(self.rotation_data) > 0:
                for rotation in self.rotation_data:
                    intersection_point, primary_support_vector, left_or_right, clfs, _ = rotation
                    print("intersection point: {}".format(intersection_point))
                    print("primary_support_vector {}".format(primary_support_vector))
                    print("Left or right: {}".format(left_or_right))

                    left_clf, right_clf = clfs

                    print("margin of left clf: {}".format(self.get_margin(left_clf)))

                    print("margin of right clf: {}".format(self.get_margin(right_clf)))
            else:
                print("Only two support vectors, no folds")




        stopper = 0


    def __init__(self,rot_func = lambda p, i, r : np.matmul(p.T - i, r) + i, max_nr_of_folds = 1, verbose = False):
        self.verbose = verbose
        self.max_nr_of_folds = max_nr_of_folds
        self.clf = svm.SVC(kernel='linear', C=1e10)
        self.old_clf = svm.SVC(kernel='linear', C=1e10)
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
