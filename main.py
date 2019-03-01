from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs


import matplotlib.pyplot as plt
import numpy as np
import math
import scipy as sci

def vector_projection(v1,v2):
    return (np.dot(v1, v2) * np.dot(v2,v2)) * v2

def group_support_vectors(support_vectors, clf):
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

def grahm_schmidt_orthonorm(linearly_independent_support_vectors):

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



def cauchy_schwarz_equal(v1,v2):
    """
    returns true if the cauchy-schwarz inequality is equal, meaning that they are linearly dependent
    if the function returns true, the vectors are linearly dependent, and one of the vectors can be removed
    """

    #inner product
    ipLeft = np.dot(v1, v2)

    return ipLeft * ipLeft == np.dot(v1,v1) * np.dot(v2,v2)
    

def find_two_linearly_independent_vectors(vectors):
    """
    Returns a matrix with the first two linearly independent vectors in it
    """

    matrix = None
    i = 0
    while i < len(vectors) - 1:

        first_vector = vectors[i]

        for second_vector in vectors[i + 1:]:

            if not(cauchy_schwarz_equal(first_vector, second_vector)):#if not equal, then they are independent.

                matrix = np.array([first_vector, second_vector])

                #exit
                i = len(vectors)
                break

        i += 1

    #if matrix == None:
     #   print("Error, no vectors are independent")

    return matrix

def find_linear_independent_vectors(vectors, matrix):
    """
    Adds linear independent vectors into a matrix recursively
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



def get_orthonormated_basis_from_support_vectors(support_vectors):

    #make the first support vector the new 'origin'
    new_origin = support_vectors[0]
    dim = len(new_origin)

    #ceate direction vectors going from the new_origin to all points
    direction_vectors = [vector - new_origin for vector in support_vectors[1:]]

    #Start with finding two linearly independent vectors of any class using cauchy schwarz inequality
    matrix = find_two_linearly_independent_vectors(direction_vectors)
    
    #add the base vectors to complement for the vectors that arn't linearly independent
    direction_vectors = np.vstack([direction_vectors, np.identity(dim)])

    #find linearly independent vectors and add them to the matrix
    matrix = find_linear_independent_vectors(direction_vectors, matrix)
    
    #create orthonormated vectors with grahm schmidt
    matrix = grahm_schmidt_orthonorm(matrix)

    
    if np.linalg.det(matrix) == 0:
        print("Error: not a basis")

    return matrix


def get_direction_between_two_vectors_in_set_with_smallest_distance(set, dim):
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

    set = np.delete(set, index_v1, 0)

    return best_dir[:dim], set

def get_align_points_rotation_matrix(direction):
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


def dimension_projection(dataset, clf):#Input: full dataset for a clf, and support vectors separated into classes in a dictionary
    #if supportvectors  -> k < n + 1 run align axis aka if(n <= currentDim) then -> align_axis
    #align_axis

    support_dict = group_support_vectors(clf.support_vectors_, clf)
    nr_of_coordinates = len(support_dict[0][0])

    while nr_of_coordinates > 2:
        
        #Rotate until we have three support vectors.
        nr_of_support_vectors = len(support_dict[0]) + len(support_dict[1])
        if nr_of_support_vectors < 3:
            return dataset, support_dict
        
        if nr_of_support_vectors == 3:
            all_support_vectors = [val for lst in support_dict.values() for val in lst]#group support vectors into one array
            rotation_matrix = get_orthonormated_basis_from_support_vectors(all_support_vectors)
            
            for i in range(0, len(dataset)):
                dataset[i][:nr_of_coordinates] = np.matmul(rotation_matrix, dataset[i][:nr_of_coordinates])
           
            for i in range(0, len(support_dict[0])):
                support_dict[0][i][:nr_of_coordinates] = np.matmul(rotation_matrix, support_dict[0][i][:nr_of_coordinates])

            for i in range(0, len(support_dict[1])):
                support_dict[1][i][:nr_of_coordinates] = np.matmul(rotation_matrix, support_dict[1][i][:nr_of_coordinates])
            
    #dataset2d = np.delete(dataset, np.s_[-dim+2], axis = 1)
    #support_dict2d = support_dict
    #support_dict2d[0] = np.delete(support_dict2d[0], np.s_[-dim+2], axis = 1)
    #support_dict2d[1] = np.delete(support_dict2d[1], np.s_[-dim+2], axis = 1)


            return dataset, support_dict


        #choose the class with most support vectors
        max_key = max(support_dict, key= lambda x: len(support_dict[x]))
        
        #get the direction between the vectors, and removes one of them from the dictionary
        direction, support_dict[max_key] = get_direction_between_two_vectors_in_set_with_smallest_distance(support_dict[max_key], rotDim)
        

        #calculate alignment matrix
        rotation_matrix = get_align_points_rotation_matrix(direction)
    

        #rotate all datapoints and support vectors
        for i in range(0, len(dataset)):
            dataset[i][:nr_of_coordinates] = np.matmul(rotation_matrix, dataset[i][:nr_of_coordinates])
           
        for i in range(0, len(support_dict[0])):
            support_dict[0][i][:nr_of_coordinates] = np.matmul(rotation_matrix, support_dict[0][i][:nr_of_coordinates])

        for i in range(0, len(support_dict[1])):
            support_dict[1][i][:nr_of_coordinates] = np.matmul(rotation_matrix, support_dict[1][i][:nr_of_coordinates])
        
        nr_of_coordinates -= 1 #exclude last coordinate

    
    return dataset, support_dict


def get_rotation(alpha):
    theta = alpha
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c,-s), (s, c)))

def rotate_point(point, angle, primary_support, intersection_point):
    """
    Returns the point rotated accordingly to rubberband folding

    Does currently not apply rubberband folding, rotates points around intersection
    """
    rotation_matrix = get_rotation(angle)

    point = np.matmul(point.T - intersection_point, rotation_matrix) + intersection_point

    return point


def rotate_set(left_clf, left_set, right_clf, right_set, primary_support):
    """
    Performs rotation on the set with biggest margin
    Currently rotates around the intersection point

    Does not contain datapoints to set

    returns a merged and rotated set, touple (X, y)
    """

    # Get margins
    right_margin = get_margin(right_clf)
    left_margin = get_margin(left_clf)

    #print("Left margin:     {}".format(left_margin))
    #print("Right margin:    {}".format(right_margin))

    # intersection data
    intersection_point, angle = get_intersection_point(left_clf, right_clf)

    # if 1, left was rotated, 0 is right set.
    left_or_right = -1

    if (right_margin > left_margin):
        #right_set[0] = removearray(right_set[0], primary_support)
        right_set[0] = [rotate_point(point, angle, primary_support, intersection_point)
                            for point in right_set[0]]
        left_or_right = 0

    elif (left_margin > right_margin):
        #left_set[0] = removearray(left_set[0], primary_support)
        left_set[0] = [rotate_point(point, -angle, primary_support, intersection_point)
                            for point in left_set[0]]
        left_or_right = 1

    else:
        print("Cannot improve margin")


    X = left_set[0] + right_set[0]

    y = left_set[1] + right_set[1]


    X = np.vstack(X)

    return (X, y, left_or_right, intersection_point)

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

def ordering_support(vectors, clf):
    """
    Returns the first possible primary support vector
    """
    primary_support_vector = None


    w = clf.coef_[0]

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

def get_splitting_point(support_dict, clf):
    """
    Finds and returns the primary support vector, splitting point
    """

    tk = ordering_support(support_dict, clf)

    first_class = tk[0][2]

    primary_support_vector = None

    for vector in tk:
        if (vector[2] is not first_class):
            primary_support_vector = vector[1]

    return primary_support_vector

def split_data(primary_support, X, Y):
    """
    returns a list  containing left and right split.
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

def plot_clf(clf, ax, XX, YY, colour='k'):
    """
    Plots a clf, with margins, colour will be black
    """

    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors=colour)

    ax.contour(XX, YY, Z, colors=colour, levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])


def plot(new_clf, old_clf, X, y):
    """
    God function that removes all the jitter from main
    """

    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    ax = plt.gca()
    xlim = (2, 10)
    ylim = (-2, -11)

    #plt.axvline(x=splitting_point, color='k')

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)

    YY, XX = np.meshgrid(yy, xx)


    plot_clf(new_clf, ax, XX, YY, 'g')
    plot_clf(old_clf, ax, XX, YY, 'k')


    plt.show()



def fold(old_clf, data_points, data_labels, support_vectors):
    # folding sets
    right_clf = svm.SVC(kernel='linear', C=1000)
    left_clf = svm.SVC(kernel='linear', C=1000)

    # Orginal support vectors
    support_dict = group_support_vectors(support_vectors, old_clf)

    # Splitting point
    primary_support_vector = get_splitting_point(support_dict, old_clf)

    # Used for plotting where the split occoured
    splitting_point = primary_support_vector[0]#x-axis-location of primary vec

    # Subsets of datasets, left and right of primary support vector
    left_set, right_set = split_data(primary_support_vector, data_points, data_labels)

    # New SVM, right

    right_clf.fit(right_set[0], right_set[1])
    left_clf.fit(left_set[0], left_set[1])

    # Rotate and merge data sets back into one
    old_X = data_points
    old_y = data_labels
    data_points, data_labels, left_or_right, intersection_point = rotate_set(left_clf,
                                                                            left_set,
                                                                            right_clf,
                                                                            right_set,
                                                                            primary_support_vector)

    # merge
    new_clf = right_clf if left_or_right else left_clf

    # Used for highlighting the sets
    right_set[0] = np.vstack(right_set[0])
    left_set[0] = np.vstack(left_set[0])


    return (new_clf, data_points, data_labels, (intersection_point,
                                                primary_support_vector,
                                                left_or_right, (right_clf, left_clf)))

def get_distance_from_line_to_point(w, point, point_on_line):
    v = point - point_on_line
    proj = vector_projection(v, w)
    distance = np.linalg.norm(v - proj)

    return distance

def clean_set(clf, data_points, data_labels):
    """
    Returns a cleaned dataset, turns soft into hard.
    Function does not calculate margin as get_margin does, something could be
    wrong with the assumption that len w is the margin. Instead it calculates
    the margin by calculating the distance from the decision_function to
    a support vector.
    """

    w = clf.coef_[0]
    w = np.array(w[1], w[0]) # place the vector in the direction of the line

    # Orginal support vectors
    support_dict = group_support_vectors(clf.support_vectors_, clf)

    point_on_line = (support_dict[0][0] + support_dict[1][0]) / 2

    margin = get_distance_from_line_to_point(w, support_dict[0][0], point_on_line)

    for point in data_points:
        distance = get_distance_from_line_to_point(w, point, point_on_line)

        if (distance < margin):
            index = np.where(data_points==point)

            data_points = np.delete(data_points, index, 0)

            data_labels = np.delete(data_labels, index)


    return (clf, data_points, data_labels)

def classify(clf, points, rotation_steps):
    for rotation in rotation_steps:
        #unpackage the mess
        intersection_point = rotation[0]
        primary_support_vector = rotation[1]
        left_or_right = rotation[2]
        right_clf = rotation[3][0]
        left_clf = rotation[3][1]

        _, angle = get_intersection_point(left_clf, right_clf)
        rotation_matrix = get_rotation(angle)

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

        rotated_set = [rotate_point(point, angle, primary_support_vector, intersection_point) for point in rotated_set]

        points = np.asarray(rotated_set + none_rotated_set)



    return clf.predict(points)

def main():
    # Dataset
    old_data_points, old_data_labels = data_points, data_labels = make_blobs(n_samples=1000,
                                                                             n_features=5,
                                                                             centers=2,
                                                                             random_state=6)

    # Original SVM
    first_clf = clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(data_points, data_labels)

    # Detect points inside the margins
    clf, old_data_points, old_data_labels = clean_set(first_clf, old_data_points,
                                                                 old_data_labels)

    #dimension reduction
    
    data_points, rotated_support_vectors = dimension_projection(data_points, clf)

    rotation_steps = []

    while(len(clf.support_vectors_) > 2):
        clf, data_points, data_labels, rotation_data = fold(clf, data_points, data_labels, support_vectors)
        rotation_steps.append(rotation_data)


    old_margin = get_margin(first_clf)
    new_margin = get_margin(clf)

    print("Old margin {}".format(old_margin))
    print("New margin post folds {}".format(new_margin))
    print("Improvement of {} and {}%".format(new_margin - old_margin, new_margin / old_margin))

    print(classify(clf, np.array([[6.4, -4.8],[5.4, -7.5]]), rotation_steps))

    plot(first_clf, clf, data_points, data_labels)
    plot(first_clf, clf, old_data_points, old_data_labels)

if __name__ == "__main__":
    main()