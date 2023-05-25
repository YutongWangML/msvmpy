import numpy as np
from numba import njit

@njit
def dual_quad_matrix(K, y, n_classes):
    # INPUT:
    # K = kernel matrix 
    #     has shape = (n_samples, n_samples)
    #     for example
    #     K[i][j] = kernel_function(X[i,:], X[j,:])
    #     
    # y = categorical labels taking value in [0, ..., num_classes - 1]
    #     y.shape = (n_samples,)
    # 
    # n_classes = number of classes
    #
    # OUTPUT:
    # Q = multiclass-ification of the kernel
    #     has shape = (n_samples*(k-1), n_samples*(k-1))
    
    # rename to look easier
    k = n_classes 
    n = len(y)
    
    Q = np.zeros((n*(k-1), n*(k-1)))
    for i in range(n):
        for j in range(n):
            subQ = np.eye(k-1) # k-1 by k-1 sub-matrix of the Q
            if y[i] == y[j]:
                subQ += 1
            elif y[i] == 0 and y[j] > 0:
                subQ[:,y[j]-1] = -1
                subQ[y[j]-1,:] = -1
                subQ[y[j]-1,y[j]-1] = -2
            elif y[i] > 0 and y[j] == 0:
                subQ[:,y[i]-1] = -1
                subQ[y[i]-1,:] = -1
                subQ[y[i]-1,y[i]-1] = -2
            else:
                subQ[:,y[i]-1] = -1
                subQ[y[j]-1,:] = -1
                subQ[y[j]-1,y[i]-1] = -2
                subQ[y[i]-1,y[j]-1] = 1

            # update Q with the sub-matrix
            Q[(i*(k-1)):((i+1)*(k-1)), (j*(k-1)):((j+1)*(k-1))] = K[i,j] * subQ
    return Q