import numpy as np

def get_algebraic_complement(matrix):
    result = np.zeros(shape=matrix.shape)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            shorten_minor = np.delete(np.delete(matrix,i,0),j,1)
            result[i][j] = np.linalg.det(shorten_minor) * (-1)**(i+j)
            
    return result