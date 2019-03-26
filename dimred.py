import numpy as np

class DR:

    def get_ungrouped_support_vectors(self, support_vectors_dictionary):
        """
        returns list of support vectors
        """
        all_support_vectors = [val for lst in support_vectors_dictionary.values() for val in lst]#group support vectors into one array
        all_support_vectors = np.stack(all_support_vectors, axis=0)

        return all_support_vectors

    def grahm_schmidt_orthonorm(self, linearly_independent_matrix):

            

            vec = linearly_independent_matrix[0]
            vec = vec / np.linalg.norm(vec)#first entry is just the itself normalized
            orthonormal_vectors = np.array([vec])#stores the new basis

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
                orthonormal_vectors = np.concatenate((orthonormal_vectors, [vec]), 0)

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

    def align(self, direction):
        #direction = np.array([0,6,6])
        
        dim = len(direction)
        matrix = np.identity(dim)

        v1 = direction[:2]
        w1 = np.linalg.norm(v1)

        if w1 != 0:#first row
            matrix[0][0] = direction[1] / w1
            matrix[0][1] = -direction[0] / w1#first subdiagonal
            
        for i in range(1, dim - 1):#middle rows
            
            v2 = direction[:i+2]
            w2 = np.linalg.norm(v2)

            if w2 > 0:
                matrix[i][i+1] = -w1 / w2#subdiagonal

            if w1 > 0:
                c2 = v2[-1]
                for k, c1 in enumerate(v1):
                    matrix[i][k] = c1 * c2 / (w1 * w2)

                v1 = v2
                w1 = w2

        if w2 > 0:#last row
            matrix[dim-1] = [c / w2 for c in direction]

        return matrix

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


    def combine_matrices(self, matrix):
        """
        n*(m+1) matrix times m*m. After multiplication current matrix will be n*m
        """

        mat = np.identity(len(self.matrices[0]))
        matrix_dim = len(matrix[0])
       
        mat[:matrix_dim,:matrix_dim] = matrix[:matrix_dim,:matrix_dim]
        
        self.matrices[self.folds_done] = np.matmul(mat, self.matrices[self.folds_done].T)
    
        return


    def transform_support_vectors(self, matrix, support_vectors_dictionary, dim):

        for key, lst in support_vectors_dictionary.items():
            support_vectors_dictionary[key] = [np.matmul(matrix, vector[:dim]) for vector in lst]



            #best_dir[:dim]


    def transform(self, matrix, data):

        return np.array([np.matmul(matrix, p) for p in data])

           
        

    
    def project_down(self, data_points, support_vectors_dictionary):

        """
        Input: All data_points, support vectors grouped into the two classes
        Output: Alla data_points, projected into a two dimensional representation of the SVM, Support dict in 2D aswell

        Projects data into two dimensions
        """
       
        nr_of_coordinates = len(support_vectors_dictionary[0][0])
        self.matrices[self.folds_done] = np.identity(nr_of_coordinates)#start with the identity
        nr_of_support_vectors = len(support_vectors_dictionary[0]) + len(support_vectors_dictionary[1])
            
        #if three or more support vectors. And less support vectors than the current dimension. Reduce using the orthonormal basis from support vectors
        if nr_of_support_vectors >= 3 and nr_of_support_vectors <= nr_of_coordinates:

            all_support_vectors = self.get_ungrouped_support_vectors(support_vectors_dictionary)
            basis_matrix = self.get_orthonormal_basis_from_support_vectors(all_support_vectors)

            #rotate data and support vectors
            #self.transform_data_and_support_vectors(basis_matrix, nr_of_coordinates)
            self.combine_matrices(basis_matrix) 

            self.transform_support_vectors(basis_matrix.T, support_vectors_dictionary, nr_of_coordinates)

            #post rotation the dimension is lowered to the number of support vectors - 1
            nr_of_coordinates = nr_of_support_vectors - 1


        #Rotate/align support vectors until we reach 2D.
        while nr_of_coordinates > 2:
            
            #choose the class with most support vectors
            max_key = max(support_vectors_dictionary, key= lambda x: len(support_vectors_dictionary[x]))
        
            #get the direction between the two support vectors, and removes one of them from the dictionary
            direction, support_vectors_dictionary[max_key] = self.get_direction_between_two_vectors_in_set_with_smallest_distance(support_vectors_dictionary[max_key], nr_of_coordinates)
        
            #calculate alignment matrix
            rotation_matrix = self.align(direction)
    
            #rotate all datapoints and support vectors
            #self.transform_data_and_support_vectors(rotation_matrix, nr_of_coordinates)
            self.transform_support_vectors(rotation_matrix, support_vectors_dictionary, nr_of_coordinates)

            self.combine_matrices(rotation_matrix)

            #support vectors are aligned.
            #exclude last coordinate for further iterations.
            nr_of_coordinates -= 1 


        data_points = self.transform(self.matrices[self.folds_done], data_points)
        

        return data_points, support_vectors_dictionary

    def project_up(self, data_points):

        """
        Use the INVERSE of the transformation matrix that projected the data into 2D
        """
        data_points = self.transform(np.linalg.inv(self.matrices[self.folds_done]), data_points)

        self.folds_done = self.folds_done + 1

        return data_points



    def __init__(self):
        self.matrices = {}
        self.folds_done = 0

        return
