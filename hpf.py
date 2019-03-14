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

        #Subtract one vector from another of the other class to get a point in the hyperplane
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

        print("Class 1: ", len(self.support_vectors_dictionary[0]), " Class  2: ", len(self.support_vectors_dictionary[1]))

    def get_ungrouped_support_vectors(self):
        """
        returns list of support vectors
        """
        all_support_vectors = [val for lst in self.support_vectors_dictionary.values() for val in lst]#group support vectors into one array
        all_support_vectors = np.stack(all_support_vectors, axis=0)
        return all_support_vectors


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

        point[:2] = self.rot_func(point[:2], intersection_point, rotation_matrix)
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
            right_set[0] = [self.rotate_point_2D(point, angle, primary_support, intersection_point) for point in right_set[0]]
            left_or_right = 0

        elif (left_margin > right_margin):
            left_set[0] = [self.rotate_point_2D(point, -angle, primary_support, intersection_point)
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
        primary_support_vector = self.get_splitting_point()

        # Used for plotting where the split occoured
        splitting_point = primary_support_vector[0]#x-axis-location of primary vec

        # Subsets of datasets, left and right of primary support vector
        left_set, right_set = self.split_data(primary_support_vector)

        # New SVM, right
        right_clf.fit(right_set[0], right_set[1])
        left_clf.fit(left_set[0], left_set[1])

        # Rotate and merge data sets back into one
        self.data_points, self.data_labels, left_or_right, intersection_point = self.rotate_set(left_clf, left_set, right_clf, right_set, primary_support_vector)

        self.rotation_data.append((intersection_point, primary_support_vector, left_or_right, (right_clf, left_clf)))

        # merge
        self.clf = right_clf if left_or_right else left_clf
        self.group_support_vectors()

        # Used for highlighting the sets
        right_set[0] = np.vstack(right_set[0])
        left_set[0] = np.vstack(left_set[0])


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

    def grahm_schmidt_orthonorm(self, linearly_independent_matrix):

        orthonormal_vectors = []#stores the new basis

        vec = linearly_independent_matrix[0]
        vec = vec / np.linalg.norm(vec)#first entry is just the itself normalized
        orthonormal_vectors.append(vec)

        i = 0
        for v in linearly_independent_matrix[1:]:

            vec = 0
            for u in orthonormal_vectors:
                projection = np.dot(v,u) * u
                projection[np.abs(projection) < 0.000001] = 0
                vec -= projection

            vec = v + vec
            vec[np.abs(vec) < 0.000001] = 0

            if all(v == 0 for v in vec):#if true then this vector is dependent on it's predicessors
                print(i)

            i += 1

            vec = vec / np.linalg.norm(vec)
            orthonormal_vectors.append(vec)

        return orthonormal_vectors


    def cauchy_schwarz_equal(self, v1, v2):
        """
        returns true if the cauchy-schwarz inequality is equal, meaning that they are linearly dependent
        if the function returns true, the vectors are linearly dependent, and one of the vectors can be removed
        """

        #inner product
        ipLeft = np.dot(v1, v2)

        return ipLeft * ipLeft == np.dot(v1,v1) * np.dot(v2,v2)

    def find_two_linearly_independent_vectors(self, vectors):
        """
        Returns a matrix with the first two linearly independent vectors in it
        """

        matrix = None
        i = 0
        while i < len(vectors) - 1:

            first_vector = vectors[i]

            for second_vector in vectors[i + 1:]:

                if not(self.cauchy_schwarz_equal(first_vector, second_vector)):#if not equal, then they are independent.

                    matrix = np.array([first_vector, second_vector])

                    #exit
                    i = len(vectors)
                    break

            i += 1

        return matrix

    def find_linear_independent_vectors(self, vectors, matrix):
        """
        fills matrix with linear independent vectors
        vectors must be complemented by appending the identity matrix for the dimension
        otherwise the matrix will not be of correct dimensions
        """
        dim = len(vectors[0])
        rank = np.linalg.matrix_rank(matrix)

        for vector in vectors:
            new_matrix = np.vstack([matrix, vector])
            new_rank = np.linalg.matrix_rank(new_matrix)

            if new_rank > rank: #if the rank is higher, the newly introduced vector is linearly independent with the vectors in the matrix, then add it to the matrix and start over with the rest of the vectors

                matrix = new_matrix#find_linear_independent_vectors(vectors[i:], new_matrix, new_rank)

                if len(matrix) == dim: 
                    if np.linalg.det(matrix) != 0: #matrix is a basis
                        return matrix
                    else:
                        print("SOMETHING WENT WRONG, Error: not a basis")
                    

                rank = new_rank
            
        return matrix

    def get_orthonormal_basis_from_support_vectors(self, support_vectors):

        #make the first support vector the new 'origin'
        new_origin = support_vectors[0]
        dim = len(new_origin)

        #ceate direction vectors going from the new_origin to all points
        direction_vectors = [vector - new_origin for vector in support_vectors[1:]]

        #Start with finding two linearly independent vectors of any class using cauchy schwarz inequality
        matrix = self.find_two_linearly_independent_vectors(direction_vectors)
    
        #add the base vectors to complement for the vectors that arn't linearly independent
        direction_vectors = np.vstack([direction_vectors, np.identity(dim)])

        #find linearly independent vectors and add them to the matrix
        matrix = self.find_linear_independent_vectors(direction_vectors, matrix)
    
        #create orthonormated vectors with grahm schmidt
        matrix = self.grahm_schmidt_orthonorm(matrix)

        return matrix

    def get_direction_between_two_vectors_in_set_with_smallest_distance(self, set, dim):
        """
        Finds the shortest distance between two vectors within the given set.
        """
        if (len(set) <2):
            print("Error, less than two support vectors in set")
            return
    
        best_dir = set[0] - set[1]
        best_dist = np.linalg.norm(best_dir)
        index_v1 = 0
        for index_v1 in range(0, len(set)):
            vec1 = set[index_v1]
            for vec2 in set[index_v1 + 1:]:

                dir = vec1 - vec2
                dist = np.linalg.norm(dir)
                if dist < best_dist:#found two vecs with shorter distance inbetween
                    best_dist = dist
                    best_dir = dir

        set = np.delete(set, index_v1, 0)#remove one of the support vectors

        return best_dir[:dim], set

    def align_direction_matrix(self, direction):
        """
        Inputs a direction, from one point to another.
        Dim, is a subdim of the total featurespace.

        Forms a lower triangular rotation matrix
        In the function, 'diagonal' is NOT denoted as the 'center'-diagonal. It is selected as: matrix[row][row+1] for a row-major matrix
        """
        dim = len(direction)
        rotation_matrix = np.zeros((dim,dim))

        #Wk = sqrt(v1^2 + v2^2 ... + vk^2) 
        squared_elements_accumulator = direction[0] * direction[0] + direction[1] * direction[1]
    
        Wk = direction[0]#for k = 1
        Wkp1 = np.sqrt(squared_elements_accumulator)
    
        #first row
        if Wkp1 != 0:
            rotation_matrix[0][0] = direction[1] / Wkp1#first element
            rotation_matrix[0][1] = -Wk / Wkp1#first diagonal element
        else:
            rotation_matrix[0][0] = 1#first element
            rotation_matrix[0][1] = 0#first diagonal element


        #middle rows
        for row in range(1, dim - 1):
        
            subdiagonal_element = direction[row + 1]#row + 1 is the k'th element in the vector
            squared_elements_accumulator += subdiagonal_element * subdiagonal_element#accumulate next step, square next element

            Wk = Wkp1
            Wkp1 = np.sqrt(squared_elements_accumulator)

        
            #subdiagonal
            U = 0
            if Wkp1 != 0:
                U = Wk / Wkp1

            rotation_matrix[row][row + 1] = -U #subdiagonal entry in matrix
             

            #denominator per row 
            denominator = Wk * Wkp1

            if denominator == 0:
                rotation_matrix[row][row] = 1

            else:
                i = 0
                for element in direction[0:row+1]:
                    rotation_matrix[row][i] = element * subdiagonal_element / denominator
                    i+=1

        #last row in matrix
        if Wkp1 != 0:
            rotation_matrix[dim-1] = [element / Wkp1 for element in direction]
        else:
            rotation_matrix[dim-1][dim-1] = 1


        return rotation_matrix

    def transform_data_and_support_vectors(self, matrix, nr_of_coordinates):

        #transform data and support vectors into the new subspace
        for i in range(0, len(self.data_points)):
            self.data_points[i][:nr_of_coordinates] = np.matmul(matrix, self.data_points[i][:nr_of_coordinates])
        
        #first class
        for i in range(0, len(self.support_vectors_dictionary[0])):
            self.support_vectors_dictionary[0][i][:nr_of_coordinates] = np.matmul(matrix, self.support_vectors_dictionary[0][i][:nr_of_coordinates])
        
        #second class
        for i in range(0, len(self.support_vectors_dictionary[1])):
            self.support_vectors_dictionary[1][i][:nr_of_coordinates] = np.matmul(matrix, self.support_vectors_dictionary[1][i][:nr_of_coordinates])

        return

    def dimension_projection(self):#Input: full dataset for a clf, and support vectors separated into classes in a dictionary
        #if supportvectors  -> k < n + 1 run align axis aka if(n <= currentDim) then -> align_axis
        #align_axis
       
        nr_of_coordinates = len(self.support_vectors_dictionary[0][0])
        nr_of_support_vectors = len(self.support_vectors_dictionary[0]) + len(self.support_vectors_dictionary[1])
            
        #if three or more support vectors. And less support vectors than the current dimension. Reduce using the orthonormal basis from support vectors
        if nr_of_support_vectors >= 3 and nr_of_support_vectors < nr_of_coordinates:

            all_support_vectors = self.get_ungrouped_support_vectors()
            basis_matrix = self.get_orthonormal_basis_from_support_vectors(all_support_vectors)

            #rotate data and support vectors
            self.transform_data_and_support_vectors(basis_matrix, nr_of_coordinates)

            #post rotation the dimension is lowered to the number of support vectors - 1
            nr_of_coordinates = nr_of_support_vectors - 1


        #Rotate/align support vectors until we reach 2D.
        while nr_of_coordinates > 2:
            
            #choose the class with most support vectors
            max_key = max(self.support_vectors_dictionary, key= lambda x: len(self.support_vectors_dictionary[x]))
        
            #get the direction between the two support vectors, and removes one of them from the dictionary
            direction, self.support_vectors_dictionary[max_key] = self.get_direction_between_two_vectors_in_set_with_smallest_distance(self.support_vectors_dictionary[max_key], nr_of_coordinates)
        
            #calculate alignment matrix
            rotation_matrix = self.align_direction_matrix(direction)
    
            #rotate all datapoints and support vectors
            self.transform_data_and_support_vectors(rotation_matrix, nr_of_coordinates)
        
            #support vectors are aligned.
            #exclude last coordinate for further iterations.
            nr_of_coordinates -= 1 


        #By now, the data should be projected into two dimensions.
        
        #save the rest of the coordinates to go back into higher dimensions when done with folding.
        #self.leftover_coordinates = [x[2:] for x in self.data_points]

        #Overwrite datapoints, with the 2D representation
       # self.data_points = [x[:2] for x in self.data_points]#2d coordinates


        return

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
        self.clf.fit(data_points, data_labels)
        self.old_clf.fit(data_points, data_labels)
        self.old_margin = self.get_margin(self.old_clf)

        #group into classes = create support_vectors_dictionary
        self.group_support_vectors()
        
        #project onto 2D
        #self.dimension_projection()

        #fold until just two support vectors exist or max_nr_of_folds is reached
        current_fold = 0
        while(len(self.clf.support_vectors_) > 2):# and current_fold < self.max_nr_of_folds):
                self.fold()
                self.new_margin = self.get_margin(self.clf)
                current_fold += 1

        stopper = 0
    
    def __init__(self, rot_func = lambda p, i, r : np.matmul(p - i, r) + i, max_nr_of_folds = 1):
        
        self.max_nr_of_folds = max_nr_of_folds
        self.clf = svm.SVC(kernel='linear', C=1000)
        self.old_clf = svm.SVC(kernel='linear', C=1000)
        self.rotation_data = []
        self.rot_func = rot_func


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
