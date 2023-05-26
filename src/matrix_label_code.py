from scipy.sparse import coo_matrix, eye
import numpy as np

def get_matrix_label_code(n_classes):
    # AMPLE: matrix-product label encoding for classification into n_classes
    
    # Without involution code:
    # R_{ij} = pi sigma_{y_i} sigma_{y_j} pi^T
    
    # With involution:
    #     R_{ij} = rho_{y_i} Theta rho_{y_j}^T
    # the above in Python is written as:
    #     self.PI[y[i]] @ self.Theta @ self.PI[y[j]].T
        
    # number of classes minus 1
    ncm1 = n_classes-1
    
    # the matrix-product label encoding matrices
    PI = [np.eye(ncm1) for j in range(n_classes)]
    for j in range(ncm1):
        PI[j][:,j] = -1

    return PI

def get_matrix_label_code_sparse(n_classes):
    # AMPLE: matrix-product label encoding for classification into n_classes
    # same as above, but using sparse encoding
    
    ncm1 = n_classes-1
    

    PI = []
    for j in range(ncm1):
        row_indices = [i for i in range(ncm1) if i != j] + [i for i in range(ncm1)]
        col_indices = [i for i in range(ncm1) if i != j] + [j for i in range(ncm1)]
        values = [1.0 for i in range(ncm1) if i != j] + [-1.0 for i in range(ncm1)]
        PI_j = coo_matrix((values,(row_indices, col_indices)))
        PI.append(PI_j)
    PI.append(eye(ncm1))    

    return PI