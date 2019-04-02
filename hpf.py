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


def plot_datapoints(data):
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

    #plt.axis([-200, 200, -200, 200])
    plt.axis([-15, 15, -15, 15])
    plt.show()

def vec_equal(vec1, vec2):
    acc = 0.0
    for elements in zip(vec1, vec2):
        acc += elements[0] - elements[1]
        
    if acc * acc < 0.00000001:
        return True

    return False


class HPF:

    def plot_plane(self, clf, new_figure_ = False):
        if new_figure_:
            plt.figure()

       # g = self.group_support_vectors(clf)
        g = clf
        h = self.get_hyperplane_direction(g)

        w = [0,0]
        w[0] = -h[1]
        w[1] = h[0]

        p = (g[0][0][:2] + g[1][0][:2]) / 2

        hx1 = p[0] + h[0] * 100
        hy1 = p[1] + h[1] * 100

        hx2 = p[0] + h[0] * -100
        hy2 = p[1] + h[1] * -100

        plt.plot([hx1,hx2],[hy1, hy2], 'b')

        hx1 = p[0] + w[0] * 100
        hy1 = p[1] + w[1] * 100

        hx2 = p[0] + w[0] * -100
        hy2 = p[1] + w[1] * -100

        plt.plot([hx1,hx2],[hy1, hy2], 'r')

       

    def plot_data_points(self, data, new_figure_ = False):

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

    def plot_data_and_plane(self):

        x1 = []
        y1 = []
        x2 = []
        y2 = []

        for index, label in enumerate(self.data[1]):

            if label == 0:

                x1.append(self.data[0][index][0])
                y1.append(self.data[0][index][1])

            elif label == 1:
                x2.append(self.data[0][index][0])
                y2.append(self.data[0][index][1])

        h = self.get_hyperplane_direction(self.support_vectors_dictionary)

        p = (self.support_vectors_dictionary[0][0][:2] + self.support_vectors_dictionary[1][0][:2]) / 2

        hx1 = p[0] + h[0] * 100
        hy1 = p[1] + h[1] * 100

        hx2 = p[0] + h[0] * -100
        hy2 = p[1] + h[1] * -100

        plt.figure()

        plt.plot(x1, y1, 'ro')
        plt.plot(x2, y2, 'go')
        plt.plot([hx1,hx2],[hy1, hy2], 'b')

    #    plt.axis([-200, 200, -200, 200])
        plt.axis([-15, 15, -15, 15])


    def vector_projection(self, v1, v2):
        """
        v1 projected on v2
        """
        return np.dot(v1, v2) / np.dot(v2,v2) * v2

    def get_hyperplane(self, clf):
        """
        Returns hyperplane for classifer
        """

        w = clf.coef_[0]
        a = -w[0] / w[1]

        return (a, (-clf.intercept_[0]) / w[1])

    def get_hyperplane_direction(self, grouped_support_vectors):
        """
        Input: Dictionary containing lists of support vectors of each class
        Output: The hyperplanes direction.
        Note, Input must have one support vector in each class.
        """

        #Remove duplicate 2d points
        if len(grouped_support_vectors) < 2:
            print("To few classes?")

        if len(grouped_support_vectors[0][0]) < 2:
            print("error, vector dimension to small")

        for lst_idx in grouped_support_vectors:
            lst = grouped_support_vectors[lst_idx]

            for n, v1 in enumerate(lst):
                
                
                for v2 in lst[n+1:]:
                    s = (v1[0] - v2[0]) + (v1[1] - v2[1])

                    if  s*s < 0.00001:
                        del lst[n]
                        print("Deleted vector")


        max_key = max(grouped_support_vectors, key= lambda x: len(grouped_support_vectors[x]))

        h = [0,0]
        normh = 1

        if len(grouped_support_vectors[max_key]) < 2: #when only one support vector in each class

            w = grouped_support_vectors[0][0][:2] - grouped_support_vectors[1][0][:2]
            h[0] = -w[1]
            h[1] = w[0]


        else:
            h = grouped_support_vectors[max_key][0][:2] - grouped_support_vectors[max_key][1][:2]


        normh = np.linalg.norm(h)

        if normh < 0.000000001:
            print("Error in get hyperplane direction asdf asdf")



        h = h / normh;#normalize
        return h

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


    def get_intersection_between_SVMs(self, left_clf, right_clf):

        left_grouped = self.group_support_vectors(left_clf)
        right_grouped = self.group_support_vectors(right_clf)


        test = (left_grouped[0][0][:2] + left_grouped[1][0][:2])

        first_left_point_on_line = (left_grouped[0][0][:2] + left_grouped[1][0][:2]) / 2
        first_right_point_on_line = (right_grouped[0][0][:2] + right_grouped[1][0][:2]) / 2

        left_direction = self.get_hyperplane_direction(left_grouped)
        right_direction = self.get_hyperplane_direction(right_grouped)

        second_left_point_on_line = first_left_point_on_line + left_direction
        second_right_point_on_line = first_right_point_on_line + right_direction

        x1 = first_left_point_on_line[0]
        x2 = second_left_point_on_line[0]
        y1 = first_left_point_on_line[1]
        y2 = second_left_point_on_line[1]

        x3 = first_right_point_on_line[0]
        x4 = second_right_point_on_line[0]
        y3 = first_right_point_on_line[1]
        y4 = second_right_point_on_line[1]


        #x = ( (x1*y2-y1*x2)*(x3-x4) - (x1-x2) *(x3*y4-y3*x4) )/ ( (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4) )
        #y = ( (x1*y2-y1*x2)*(y3-y4) - (y1-y2) *(x3*y4-y3*x4) )/ ( (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4) )

        cosang = np.dot(right_direction, left_direction)

        angle = math.acos(cosang)*180 /3.1415

        print(angle)

        #cosang = math.sqrt((1 + cosang) / 2)
        angle2 = math.acos(cosang)*180 / 3.1415
        print(angle2)

        #return [x, y], np.abs(cosang)


        #print(angle*180 /3.1415)

        #left_slope = (second_left_point_on_line[1] - first_left_point_on_line[1]) / (second_left_point_on_line[0] - first_left_point_on_line[0])
        #right_slope = (second_right_point_on_line[1] - first_right_point_on_line[1]) / (second_right_point_on_line[0] - first_right_point_on_line[0])


        xy, angle = self.get_intersection_point(left_clf, right_clf)

        return [xy[0], xy[1]], -angle




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
        h = self.get_hyperplane_direction(self.support_vectors_dictionary)

        # As normal for the line is W = (b, -a)
        # direction is then given by as a = (-(-a), b))

        a = h[0]
        b = h[1]

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


    def left_or_right_of_plane(self, point):
        """
        Splits the data from the primary point in the direction of the normal
        """
        #w = self.clf.coef_[0]

        

        h = self.get_hyperplane_direction(self.support_vectors_dictionary)


        v = point[:2] - self.primary_support_vector[:2]#direction from one vector to the splitting point
        normv = np.linalg.norm(v)
        if normv < 0.000000001: #same point as primary support vector in 2d. Add into left set? or both? TODO:
            print("same as primary")
            return 0

        v = v / normv

        cosang = np.dot(h,v)#since both are normalized, according to dot products definition, returns the cosine of the angle between the directions.

        if cosang > 0:#Left / right
            return 1
        else:
            return 0

    

    def split_data(self):
        """
        returns a list  containing left and right split.
        """
        # Construct a new array, to remove reference

        right_set = [[],[]]
        left_set = [[],[]]

        right_set_2d = [[],[]]
        left_set_2d = [[],[]]

        for vector in zip(self.data[0], self.data[1]):
            vector_2d = (vector[0][:2], vector[1])

            #Uses failchecks to NOT add any duplicates. One of the aligned vectors needs to be excluded from training
            if vec_equal(vector[0], self.primary_support_vector):#all(vector[0] == self.primary_support_vector):#is primary support vector
                right_set[0].append(np.array(vector[0]))
                right_set[1].append(vector[1])

                left_set[0].append(np.array(vector[0]))
                left_set[1].append(vector[1])

            if vec_equal(vector_2d[0], self.primary_support_vector[:2]):#is primary support vector
                if not any(vec_equal(vector_2d[0], x) for x in right_set_2d[0]):
                    right_set_2d[0].append(np.array(vector_2d[0]))
                    right_set_2d[1].append(vector_2d[1])
                if not any(vec_equal(vector_2d[0], x) for x in left_set_2d[0]):
                    left_set_2d[0].append(np.array(vector_2d[0]))
                    left_set_2d[1].append(vector_2d[1])
            else:

                if self.left_or_right_of_plane(vector[0]):
                    right_set[0].append(np.array(vector[0]))
                    right_set[1].append(vector[1])

                    if not any(vec_equal(vector_2d[0], x) for x in right_set_2d[0]):
                        right_set_2d[0].append(np.array(vector_2d[0]))
                        right_set_2d[1].append(vector_2d[1])
                else:
                    left_set[0].append(np.array(vector[0]))
                    left_set[1].append(vector[1])
                    if not any(vec_equal(vector_2d[0], x) for x in left_set_2d[0]):
                        left_set_2d[0].append(np.array(vector_2d[0]))
                        left_set_2d[1].append(vector_2d[1])




        return left_set, left_set_2d, right_set, right_set_2d

    def get_splitting_point(self):
        """
        Finds and returns the primary support vector, splitting point
        """

        tk = self.ordering_support()
        first_class = tk[0][2]


        for vector in tk:
            if (vector[2] is not first_class):
                return vector[1]

        print("No primary vector found")

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
                grouped_support_vectors[key].append(sv)

        return grouped_support_vectors

    def get_rotation(self, alpha):
        """
        Forms the rotation matrix:
        (cos, sin)
        (-sin, cos)
        """
        theta = alpha
        c, s = np.cos(theta), np.sin(theta)
        return np.array(((c,-s), (s, c)))

    def get_rotation_a(self, cos_angle):
        """
        Forms the rotation matrix:
        (cos, sin)
        (-sin, cos)
        """
        sin_angle = math.sin(math.acos(cos_angle))

        if cos_angle > 0:
            sin_angle = -sin_angle

        

        test = math.asin(sin_angle)
        test2 = math.acos(cos_angle)


        return np.array(((cos_angle,-sin_angle), (sin_angle, cos_angle)))



    def rotate_point_2D(self, point, angle, primary_support, intersection_point):
        """
        Returns the point, with it's xy-coordinates rotated accordingly to rubberband folding

        Does currently not apply rubberband folding, rotates points around intersection
        """
        rotation_matrix = self.get_rotation(angle)

        #point = np.matmul(point.T - intersection_point, rotation_matrix) + intersection_point

        #Slica föri helvete inte point i tilldelning här
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
        intersection_point, cos_angle = self.get_intersection_between_SVMs(left_clf, right_clf)#self.get_intersection_point(left_clf, right_clf)
        # if 1, left was rotated, 0 is right set.
        left_or_right = -1



        if (right_margin >= left_margin):
            right_set[0] = [self.rotate_point_2D(point, -cos_angle, primary_support_vector,
                            intersection_point)
                                for point in right_set[0]]
            left_or_right = 0

        elif (left_margin > right_margin):
            left_set[0] = [self.rotate_point_2D(point, cos_angle, primary_support_vector,
                            intersection_point)
                                for point in left_set[0]]
            left_or_right = 1





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
        
        # Subsets of datasets, left and right of primary support vector

        #if self.current_fold > 30:
         #   self.plot_plane(self.support_vectors_dictionary, True)
          #  self.plot_data_points(self.data)
           # plt.show()
            #stopper = 0

       

        left_set, left_set_2d, right_set, right_set_2d = self.split_data()

        #self.plot_data_and_plane()


        #self.plot_data_points(right_set_2d)
        #plt.show()
        # New SVM, right
        try:
            right_clf.fit(right_set_2d[0], right_set_2d[1])
            left_clf.fit(left_set_2d[0], left_set_2d[1])
        except ValueError:
            print("WARNING, ONLY ONE CLASS PRESENT IN A SET, ABORTING")
            return -1

        #if self.current_fold > 30:
         #   print(self.primary_support_vector)
          #  self.plot_plane(self.support_vectors_dictionary, True)
           # self.plot_data_points(self.data)
            
            #self.plot_plane(right_clf, True)
            #self.plot_plane(left_clf)
           # self.plot_data_points(self.data)
           # plt.show()

        #self.plot_plane(right_clf, True)
        #self.plot_data_points(left_set_2d)
        #self.plot_data_points(right_set_2d)
        #self.plot_plane(left_clf)
        #plt.axis([-15, 15, -15, 15])

        #plt.axis([-15, 15, -15, 15])
        #plt.show()

        #self.plot_plane(right_clf, True)
        #self.plot_plane(left_clf)
        #self.plot_data_points(self.data)
        #plt.axis([-15, 15, -15, 15])
        

        # Rotate and merge data sets back into one
        self.data[0], self.data[1], left_or_right, intersection_point = self.rotate_set(left_clf, left_set, right_clf, right_set, self.primary_support_vector)


        #if self.current_fold > 30:
         #   self.plot_plane(self.support_vectors_dictionary, True)
          #  self.plot_data_points(self.data)
            #plt.show()


        #self.plot_plane(right_clf, True)
        #self.plot_plane(left_clf)
        #self.plot_data_points(self.data)
        #plt.axis([-15, 15, -15, 15])
        #plt.show()

        #self.plot_plane(right_clf, True)
        #self.plot_plane(left_clf)
        #self.plot_data_points(self.data)
        #plt.axis([-15, 15, -15, 15])
        #4plt.show()


        self.rotation_data.append((intersection_point, self.primary_support_vector, left_or_right, (right_clf, left_clf)))

        # merge
        #self.clf = right_clf if left_or_right else left_clf
        #self.group_support_vectors(self.clf)

        # Used for highlighting the sets
        #right_set[0] = np.vstack(right_set[0])
        #left_set[0] = np.vstack(left_set[0])
        return 0

    def classify(self, points, rotate=True):
        if not rotate:
            return self.old_clf.predict(points)

        for idx, rotation in enumerate(self.rotation_data):

            points = self.dim_red.classify_project_down(points, idx)

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
                    if self.left_or_right_of_plane(point, primary_support_vector):
                        rotated_set.append(point)
                    else:
                        none_rotated_set.append(point)
                else:
                    if not self.left_or_right_of_plane(point, primary_support_vector):
                        rotated_set.append(point)
                    else:
                        none_rotated_set.append(point)

            rotated_set = [self.rotate_point_2D(point, angle, primary_support_vector, intersection_point) for point in rotated_set]

            points = np.asarray(rotated_set + none_rotated_set)

            points = self.dim_red.classify_project_up(points, idx)

        return self.clf.predict(points)




    def fit(self, data_points, data_labels):
        #self.data_points = data_points
        #self.data_labels = data_labels
        self.data = [data_points, data_labels]

        self.old_data = data_points


        self.old_clf.fit(data_points, data_labels)
        self.old_margin = self.get_margin(self.old_clf)
        self.new_margin = -1


      #  plot_datapoints(self.data, self.clf)
        #project onto 2D
        self.current_fold = 0
        val = 0
        self.clf.fit(data_points, data_labels)
        self.support_vectors_dictionary = self.group_support_vectors(self.clf) #group into classes = create support_vectors_dictionary

        previous_margin = self.old_margin
        margins = []
        while(len(self.clf.support_vectors_) > 2 and val is 0):


            self.data[0], self.support_vectors_dictionary = self.dim_red.project_down(self.data[0], self.support_vectors_dictionary)

            #plot_datapoints(self.data)

           # plot_datapoints(self.data)
            #fold until just two support vectors exist or max_nr_of_folds is reached
            #self.plot_data_and_plane()
            val = self.fold()

            self.current_fold += 1
    #        plot_datapoints(self.data)

            self.data[0] = self.dim_red.project_up(self.data[0])



#            plot_datapoints(self.data)
            self.clf.fit(self.data[0], self.data[1])#fit for next iteration or exit contidion of just two support vectors
            self.support_vectors_dictionary = self.group_support_vectors(self.clf) #regroup

            

            #self.plot_data_and_plane()
            #plt.show()
            self.new_margin = self.get_margin(self.clf)
            margins.append(math.fabs(self.new_margin - previous_margin))
            previous_margin = self.new_margin

            if len(margins) == 10:
                avg = sum(margins) / len(margins)
                if avg < 0.00001:
                    val = -1
                    print("termination due to floating point")
                margins.clear()


        print(self.new_margin)
        print(self.old_margin)

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
                    intersection_point, primary_support_vector, left_or_right, clfs = rotation
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
