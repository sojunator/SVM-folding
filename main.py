from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs


import matplotlib.pyplot as plt
import numpy as np
import math
import scipy as sci

def vector_projection(v1,v2):
    dot1 = np.dot(v1,v2)
    dot2 = np.dot(v2,v2)
    return (np.dot(v1, v2) / np.dot(v2,v2)) * v2

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
    


def find_linear_independent_vectors(vectors, matrix, rank):
    """
    Adds linear independent vectors into a matrix recursively
    """

    i = 0
    while i < len(vectors):

        vector = vectors[i]
        new_matrix = np.vstack([matrix, vector])
        new_rank = np.linalg.matrix_rank(new_matrix)

        if new_rank > rank: #if the rank is higher, the newly introduced vector is linearly independent with the vectors in the matrix, then add it to the matrix and start over

            #vectors = np.delete(vectors, i, 0)#remove the vector from the mass that is linearly independent

            matrix = find_linear_independent_vectors(vectors[i:], new_matrix, new_rank)
            
            #exit
            i = len(vectors)

        i += 1

    return matrix


def get_orthonormated_basis_from_support_vectors(support_vectors):

    #linearly_independent_support_vectors = support_vectors
    
    #Start with finding two linearly independent supportvectors of any class using cauchy schwarz inequality
    matrix = None
    i = 0
    while i < len(support_vectors) - 1:

        firstVector = support_vectors[i]

        for secondVector in support_vectors[i + 1:]:

            if not(cauchy_schwarz_equal(firstVector, secondVector)):

                matrix = np.array([firstVector, secondVector])

                #exit
                i = len(support_vectors)
                break

        i += 1

    #Store the rank of the current matrix
    rank = np.linalg.matrix_rank(matrix)
    
    matrix = find_linear_independent_vectors(support_vectors, matrix, rank)
    
    
    

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

def align_axis(support_vectors):

    linearly_independent_support_vectors = 0

    orthonormated_basis = grahm_schmidt_orthonorm(linearly_independent_support_vectors)

    return

def get_direction_between_two_vectors_in_set_with_smallest_distance(set, dim):
    """
    Finds the shortest distance between two vectors within the given set.
    """
    if (len(set) <2):
        print("Error, less than two support vectors in set")
        return
    
    bestDir = set[0] - set[1]
    bestDist = np.linalg.norm(bestDir)
    index_v1 = 0
    for index_v1 in range(0, len(set)):
        vec1 = set[index_v1]
        for vec2 in set[index_v1 + 1:]:

            dir = vec1 - vec2
            dist = np.linalg.norm(dir)
            if dist < bestDist:#found two vecs with shorter distance inbetween
                bestDist = dist
                bestDir = dir

    set = np.delete(set, index_v1, 0)

    return bestDir[:dim], set

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
    squaredElementsAccumulator = direction[0] * direction[0] + direction[1] * direction[1]
    
    Wk = direction[0]#for k = 1
    Wkp1 = np.sqrt(squaredElementsAccumulator)
    
    #first row
    if Wkp1 != 0:
        rotation_matrix[0][0] = direction[1] / Wkp1#first element
        rotation_matrix[0][1] = -Wk / Wkp1#first diagonal element
    else:
        rotation_matrix[0][0] = 1#first element
        rotation_matrix[0][1] = 0#first diagonal element

   
    #middle rows
    for row in range(1, dim - 1):
        
        diagonalElement = direction[row + 1]#row + 1 is the k'th element in the vector
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
            for element in direction[0:row+1]:
                rotation_matrix[row][i] = element*diagonalElement / denominator
                i+=1
    
    #last row in matrix
    if Wkp1 != 0:
        rotation_matrix[dim-1] = [element / Wkp1 for element in direction]
    else:
        rotation_matrix[dim-1][dim-1] = 1


    return rotation_matrix


def dimension_projection(dataset, support_dict):#Input: full dataset for a clf, and support vectors separated into classes in a dictionary
    #if supportvectors  -> k < n + 1 run align axis aka if(n <= currentDim) then -> align_axis
    #align_axis

    rotDim = dim = len(support_dict[0][0])

    while rotDim > 2:
        support_vectors_from_one_class = 0
        if (len(support_dict[0]) > len(support_dict[1])):#pick class with most vectors in
            support_vectors_from_one_class = support_dict[0]
        else:
            support_vectors_from_one_class = support_dict[1]

        
        max_key = max(support_dict, key= lambda x: len(support_dict[x]))

        if len(support_dict[max_key]) < 2:
            stopper = 0
            break

        direction, support_dict[max_key] = get_direction_between_two_vectors_in_set_with_smallest_distance(support_dict[max_key], rotDim)

        rotation_matrix = get_align_points_rotation_matrix(direction)
    


        #rotate all datapoints
        for i in range(0, len(dataset)):
            dataset[i][:rotDim] = np.matmul(rotation_matrix, dataset[i][:rotDim])
           
        for i in range(0, len(support_dict[0])):
            support_dict[0][i][:rotDim] = np.matmul(rotation_matrix, support_dict[0][i][:rotDim])

        for i in range(0, len(support_dict[1])):
            support_dict[1][i][:rotDim] = np.matmul(rotation_matrix, support_dict[1][i][:rotDim])
        
        rotDim -= 1

    dataset2d = np.delete(dataset, np.s_[-dim+2], axis = 1)
    support_dict2d = support_dict
    support_dict2d[0] = np.delete(support_dict2d[0], np.s_[-dim+2], axis = 1)
    support_dict2d[1] = np.delete(support_dict2d[1], np.s_[-dim+2], axis = 1)


    return dataset, support_dict, dataset2d, support_dict2d


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
        right_set[0] = [rotate_point(point, angle, primary_support, intersection_point)
                            for point in right_set[0]]
        left_or_right = 0

    elif (left_margin > right_margin):
        left_set[0] = [rotate_point(point, -angle, primary_support, intersection_point)
                            for point in left_set[0]]

        left_or_right = 1

    else:
        print("Cannot improve margin")

    X = left_set[0] + right_set[0]
    y = left_set[1] + right_set[1]

    X = np.vstack(X)

    return (X, y, left_or_right)

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


def plot(new_clf, old_clf, X, y, splitting_point):
    """
    God function that removes all the jitter from main
    """

    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    ax = plt.gca()
    xlim = (2, 10)
    ylim = (-2, -11)

    plt.axvline(x=splitting_point, color='k')

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)

    YY, XX = np.meshgrid(yy, xx)


    plot_clf(new_clf, ax, XX, YY, 'g')
    plot_clf(old_clf, ax, XX, YY, 'k')


    plt.show()

def test_get_rotation_matrix_onto_lower_dimension():

    test_vectors = np.array([[1,0,0],[2,0,0]])
    test_ans = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])
    rotation_matrix = get_rotation_matrix_onto_lower_dimension(test_vectors, 3)
    if np.array_equal(rotation_matrix, test_ans) == False:
        print("Error, wrong matrix")
        
    test_vectors = np.array([[0,0,1],[0,0,2]])
    test_ans = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    rotation_matrix = get_rotation_matrix_onto_lower_dimension(test_vectors, 3)
    if np.array_equal(rotation_matrix, test_ans) == False:
        print("Error, wrong matrix")
        
    test_vectors = np.array([[0,1,0],[0,2,0]])
    test_ans = np.array([[-1.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]])
    rotation_matrix = get_rotation_matrix_onto_lower_dimension(test_vectors, 3)
    if np.array_equal(rotation_matrix, test_ans) == False:
        print("Error, wrong matrix")
        

    return

def test_get_direction_between_two_vectors_in_set_with_smallest_distance():
    testVectors = np.array([[2,2,2,2,2], [1,3,3,5,5], [2,0,0,0,2], [3,0,0,0,2], [4,6,4,-4,-4], [5,2,-6,-7,2],[1,2,3,2,1]])

    ans = np.array([-1,0,0,0,0])

    dir = get_direction_between_two_vectors_in_set_with_smallest_distance(testVectors)


    if all(dir) != all(ans):
        print("Error, wrong direction, Expected: ", ans, "Recieved: ", dir)
    else:
        print("Test passed")
    

   #print(rotation_matrix_onto_lower_dimension(testVectors))
  
def get_test_rot_data():
    #data_points = np.array([[1.5, 1.5, 1.0, 1.0], [0.5, 1.5, 1.0, 1.1], [1.6, 1.6, 1.0, 1.2], [3.0, 1.5, 1.0, 1.3], [0.0, 1.5, 3.5,1.4], [1.0, 2.0, 3.2, 1.5], [1.0, -1.0, -3.05,-2], [1.0, -1.5, -4.55,-2.5], [1.5, -1.2, -4,-3]])
    data_points = np.array([[1.5, 1.5, 1.0], [0.5, 1.5, 1.0], [1.6, 1.6, 1.0], [3.0, 1.5, 1.0], [0.0, 1.5, 3.5], [1.0, 2.0, 3.2], [1.0, -1.0, -3.05], [1.0, -1.5, -4.55], [1.5, -1.2, -4]])
    labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])

    return data_points, labels

def test_rot():

    data, labels = get_test_rot_data()

    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(data, labels)

    support_dict = group_support_vectors(clf.support_vectors_, clf)

    #2D align
    data, support_dict = dimension_reduction(data, support_dict)

    plot(clf, clf, data, labels)


def asdf():

    testVectors = np.array([[0,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,1,0],[1,1,0,0],[0,0,0,1]])

    asdfasdf = get_orthonormated_basis_from_support_vectors(testVectors)

    r = np.linalg.matrix_rank(testVectors)
    kekMat = sci.linalg.orth(testVectors)

    stopper = 0


def main():
    # Dataset
    
   # data_points, data_labels = make_blobs(n_samples=30, n_features=2, centers=2, random_state=6)
    data_points, data_labels = get_test_rot_data()
    
    # Original SVM
    old_clf = svm.SVC(kernel='linear', C=10000)

    # folding sets
    right_clf = svm.SVC(kernel='linear', C=10000)
    left_clf = svm.SVC(kernel='linear', C=10000)

    # Train on inital data
    old_clf.fit(data_points, data_labels)

    old_margin = get_margin(old_clf)


    # Orginal support vectors
    support_dict = group_support_vectors(old_clf.support_vectors_, old_clf)

    #2D align
    data_pointshd, support_dicthd, data_points, support_dict = dimension_projection(data_points, support_dict)



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
    data_points, data_labels, left_or_right = rotate_set(left_clf, left_set, right_clf, right_set, primary_support_vector)

    # merge
    new_clf = right_clf if left_or_right else left_clf
    """
    new_clf = svm.SVC(kernel="linear", C=10000)
    new_clf.fit(X, y)
    """
    new_margin = get_margin(new_clf)

    # Used for highlighting the sets
    right_set[0] = np.vstack(right_set[0])
    left_set[0] = np.vstack(left_set[0])


    # plot new clf (post hyperplane folding) and old clf.
    # Blue is old, red is new.
    plot(old_clf, new_clf, data_points, data_labels, splitting_point)

if __name__ == "__main__":
    asdf()
