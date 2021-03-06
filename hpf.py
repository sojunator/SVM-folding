from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.datasets import make_blobs


import matplotlib.pyplot as plt
import numpy as np
import math
import json
import pdb
import datetime


from dimred import DR

C_param = 1e10

def plot_2d(data, support_vector_dict = None):

    plt.figure()

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


    if support_vector_dict is not None:
        for key in support_vector_dict:
            for sv in support_vector_dict[key]:
                plt.plot(sv[0], sv[1], 'bx')

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

def vec_equal(vec1, vec2):

    return np.allclose(vec1, vec2)

def plot_clf(data):
    plt.figure()

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

    def plot_dir(self, dir, point, new_figure_ = False, col = 'b'):
        if new_figure_:
            plt.figure()

        h = dir

        w = [0,0]
        w[0] = -h[1]
        w[1] = h[0]

        p = point

        plot_size = 1

        hx1 = p[0] + h[0] * plot_size
        hy1 = p[1] + h[1] * plot_size

        hx2 = p[0] + h[0] * -plot_size
        hy2 = p[1] + h[1] * -plot_size

        plt.plot([hx1,hx2],[hy1, hy2], col)


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


    def plot_primary(self):

        plt.plot(self.primary_support_vector[0], self.primary_support_vector[1], 'bx')

    def plot_self(self, new_figure = False):
        if new_figure:
            plt.figure()


        p = (self.support_vectors_dictionary[0][0][:2] + self.support_vectors_dictionary[1][0][:2]) / 2

        n = self.hyperplane_normal
        h = [0,0]
        h[1] = n[0]
        h[0] = -n[1]

        self.plot_dir(h, p)

        self.plot_data(self.data)




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


    def left_or_right_of_plane(self, point, primary_support_vector = None, hyperplane_normal = None):
        """
        Splits the data from the primary point in the direction of the normal
        """

        # Classify needs to overwrite sv
        if primary_support_vector is None and hyperplane_normal is None:
            primary_support_vector = self.primary_support_vector[:2]
            hyperplane_normal = self.hyperplane_normal

        n = hyperplane_normal

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

    def split_data(self, data = None, labels = None, primary_support_vector = None, hyperplane_normal = None):
        """
        returns a list  containing left and right split.
        """
        # Construct a new array, to remove reference
        if (data is None and primary_support_vector is None):
            data = np.array(self.data[0])
            primary_support_vector = self.primary_support_vector
            labels = self.data[1]
            hyperplane_normal = self.hyperplane_normal

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
                if self.left_or_right_of_plane(point, primary_support_vector[:2], hyperplane_normal):
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
        alpha = np.min((alpha, 1.0))
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
        alpha = np.min((alpha, 1.0))
        c = alpha

        s = np.sqrt(1 - alpha * alpha)
        #theta = alpha
        #c, s = np.cos(theta), np.sin(theta)
        return np.array(((c,s), (-s, c)))


    def get_rubber_band_angle(self, point, angle, intersection_point, right_normal):


        v = point[:2] - intersection_point
        v = v / np.linalg.norm(v)

        r_angle = np.min(np.dot(v, right_normal))
        

        return np.fmax(r_angle, angle)



    def rotate_left(self, point, angle, intersection_point, right_normal, use_rubber_band):

        if use_rubber_band:
            angle = self.get_rubber_band_angle(point, angle, intersection_point, right_normal)

        rotation_matrix = self.get_counter_rotation(angle)

        x,y = self.rot_func(point[:2], intersection_point, rotation_matrix)

        point[0] = x
        point[1] = y

        self.temp_angle = angle
        return point

    def rotate_right(self, point, angle, intersection_point, right_normal, use_rubber_band):

        if use_rubber_band:
            angle = self.get_rubber_band_angle(point, angle, intersection_point, right_normal)

        rotation_matrix = self.get_rotation(angle)

        x,y = self.rot_func(point[:2], intersection_point, rotation_matrix)

        point[0] = x
        point[1] = y
        self.temp_angle = angle
        return point



    def rotate_set(self, left_clf, right_clf, primary_support_vector = None, hyperplane_normal = None, points = None, use_rubber_band = True):
        """
        Performs rotation on the set with biggest margin
        Currently rotates around the intersection point

        Does not contain datapoints to set

        returns a merged and rotated set, touple (X, y)
        """
        if primary_support_vector is None and hyperplane_normal is None and points is None:
            hyperplane_normal = self.hyperplane_normal
            primary_support_vector = self.primary_support_vector
            points = self.data

        # intersection data
        intersection_point, angle = self.get_intersection_between_SVMs(left_clf, right_clf)#self.get_intersection_point(left_clf, right_clf)


        left_or_right = -1

        #Split data based on the normal
        rotate_set = []
        non_rotate_set = []


        right_normal = right_clf.coef_[0] / np.linalg.norm(right_clf.coef_[0])#plane
        right_plane = [right_normal[1], -right_normal[0]]

        left_normal = left_clf.coef_[0] / np.linalg.norm(left_clf.coef_[0])#plane
        left_plane = [left_normal[1], -left_normal[0]]


        #plot_clf(self.data)
        #self.plot_primary()
        #self.plot_dir(left_plane, intersection_point)
        #self.plot_dir(right_plane, intersection_point, False, 'g')
        #self.plot_dir(left_normal, intersection_point, False, 'r')




        for point in zip(points[0], points[1]):

            p_i = point[0][:2] - intersection_point
            p_i = p_i / np.linalg.norm(p_i)

            if np.dot(p_i, right_normal) > 0.0:
                if np.dot(p_i, left_plane) > 0.0:
                    rotate_set.append(np.array(point))
                else:
                    non_rotate_set.append(np.array(point))
            else:
                non_rotate_set.append(np.array(point))


        #rotate clockwise or counter clockwise
        d = primary_support_vector[:2] - intersection_point
        d = d / np.linalg.norm(d)

        n = hyperplane_normal[:2]
        n = n / np.linalg.norm(n)

        c_or_cc = np.dot(d, n)#if the cosine of the angle is less than 0, rotate left.

        if c_or_cc > 0.0:
            rotate_set = [(self.rotate_left(point[0], angle, intersection_point, left_normal, use_rubber_band), point[1]) for point in rotate_set]
            left_or_right = 1

        else:
            rotate_set = [(self.rotate_right(point[0], angle, intersection_point, left_normal, use_rubber_band), point[1]) for point in rotate_set]
            left_or_right = 0


        tup = rotate_set + non_rotate_set

        self.data[0] = [p[0] for p in tup]
        self.data[1] = [p[1] for p in tup]


        #plot_clf(self.data)
        #self.plot_dir(left_plane, intersection_point)
        #self.plot_dir(right_plane, intersection_point, False, 'g')
        #self.plot_dir(left_normal, intersection_point, False, 'r')
        #self.plot_primary()
        #plt.show()

        #self.plot_dir(right_normal, intersection_point, False, True)
       # self.plot_dir(left_normal, intersection_point, True, True)
       # self.plot_data(self.data)

        #plt.show()
        return left_or_right, intersection_point


    def fold(self):

        # folding sets
        right_clf = svm.SVC(kernel='linear', C=C_param, cache_size = 7000)
        left_clf = svm.SVC(kernel='linear', C=C_param, cache_size = 7000)


        # Splitting point
        self.primary_support_vector = self.get_splitting_point()


        # Subsets of datasets, left and right of primary support vector
        right_set, left_set = self.split_data()


        # New SVM, right
        try:
            #Reduce problem down to 2d
            right_2d = [np.array(point[:2]) for point in right_set[0]]
            left_2d = [np.array(point[:2]) for point in left_set[0]]


            right_clf.fit(right_2d, right_set[1])
            left_clf.fit(left_2d, left_set[1])

        except ValueError:
            print("WARNING, ONLY ONE CLASS PRESENT IN A SET, ABORTING")
            return -1

        

        # Rotate and merge data sets back into one
        left_or_right, intersection_point = self.rotate_set(left_clf, right_clf)

        

        self.rotation_data.append((intersection_point, self.primary_support_vector, left_or_right, (right_clf, left_clf), self.support_vectors_dictionary, np.array(self.hyperplane_normal), self.temp_angle))

        return 0

    def classify(self, points, rotate=True):
        if (len(points) is 0):
            return []
        if not rotate:
            labels = self.old_clf.predict(points)

            

            #plot_3d([points, labels], np.array(self.old_clf.coef_[0]), self.old_clf.intercept_[0])
            #plt.show()

            return labels

        # Correct dim is used to save last down projection
        # As we don't want if-statement checking for last iteration
        correct_dim = []
        for idx, rotation in enumerate(self.rotation_data):

            points = self.dim_red.classify_project_down(points, idx)



            #unpackage the mess
            primary_support_vector = rotation[1]
            right_clf = rotation[3][0]
            left_clf = rotation[3][1]
            hyperplane_normal = rotation[5]

            #left_set, right_set = self.split_data(points, [None] * len(points), primary_support_vector, hyperplane_normal)


            self.rotate_set(left_clf, right_clf, primary_support_vector, hyperplane_normal, (points, [None] * len(points)), True)

            points = self.dim_red.classify_project_up(points, idx)


        return self.clf.predict(points)


    def fit(self, data_points, data_labels, time_dict):
        self.data = [data_points, data_labels]
        self.fitting = True
        self.old_data = data_points

        # Builds clfs


        svm_start_time = datetime.datetime.now()

        self.old_clf.fit(data_points, data_labels)
        svm_fit_time = datetime.datetime.now() - svm_start_time
        time_dict = (svm_fit_time.total_seconds()*1000)
        self.clf.fit(data_points, data_labels)

        print("Before folds: nr of support {}".format(len(self.clf.support_vectors_)))

        self.old_margin = self.get_margin(self.old_clf)
        self.new_margin = -1

        
        plot_3d([self.data[0], self.data[1]], np.array(self.clf.coef_[0]), self.clf.intercept_[0])
       


        #project onto 2D
        self.current_fold = 0
        val = 0
        #group into classes = create support_vectors_dictionary
        self.support_vectors_dictionary = self.group_support_vectors(self.clf)
        self.hyperplane_normal = self.get_hyperplane_normal()

        # floating point termination
        previous_margin = self.old_margin
        margins = []
        #print(self.data[0][-5:])
        while(len(self.clf.support_vectors_) > 2 and val is 0 and self.current_fold < self.max_nr_of_folds):

            #print(self.current_fold)

            self.data[0], self.support_vectors_dictionary, self.hyperplane_normal = self.dim_red.project_down(self.data[0], self.support_vectors_dictionary, self.hyperplane_normal)
            plot_2d(self.data, self.support_vectors_dictionary)
            val = self.fold()

            self.current_fold += 1

            self.data[0] = self.dim_red.project_up(self.data[0])
            #print(self.current_fold)

            #fit for next iteration or exit contidion of just two support vectors
            self.clf.fit(self.data[0], self.data[1])

            self.support_vectors_dictionary = self.group_support_vectors(self.clf) #regroup
            self.hyperplane_normal = self.get_hyperplane_normal()

            self.new_margin = self.get_margin(self.clf)
            margins.append(math.fabs(self.new_margin - previous_margin))
            previous_margin = self.new_margin
            """
            if len(margins) == 3:
                avg = sum(margins) / len(margins)
                if avg < 0.01:
                    val = -1
                    print("termination due to floating point")
                margins.clear()
            """
            stopper = None

        self.fitting = False
        print("Current fold", self.current_fold)
        print("After folds: nr of support {}".format(len(self.clf.support_vectors_)))
        plot_3d([self.data[0], self.data[1]], np.array(self.clf.coef_[0]), self.clf.intercept_[0])
        plt.show()

        #self.clf.fit(self.data_points, self.data_labels)
        #self.new_margin = self.get_margin(self.clf)
        self.new_margin = self.get_margin(self.clf)
        #print(self.new_margin)
        #print(self.old_margin)

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




        return self.old_margin, self.new_margin, time_dict


    def __init__(self,rot_func = lambda p, i, r : np.matmul(p.T - i, r) + i, max_nr_of_folds = 1, verbose = False):
        self.verbose = verbose
        self.max_nr_of_folds = max_nr_of_folds
        self.clf = svm.SVC(kernel='linear', C=C_param, cache_size = 7000)
        self.old_clf = svm.SVC(kernel='linear', C=C_param, cache_size = 7000)
        self.rotation_data = []
        self.rot_func = rot_func
        self.dim_red = DR()
        self.temp_angle = 0

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
